from abc import ABC, abstractmethod
import torch
from torch import nn
import firedrake as fd
from firedrake.ml.pytorch.fem_operator import fem_operator
from firedrake.adjoint import Control, ReducedFunctional

class BaseFiredrakeOperator(nn.Module, ABC):
    """
    Generic PyTorch wrapper for a Firedrake operator using:
        pyadjoint.ReducedFunctional + firedrake.ml.pytorch.fem_operator

    Subclasses only need to define:
    - mesh construction
    - function space construction
    - problem setup
    - annotated solve
    - optional input preprocessing
    """
    def __init__(self):
        super().__init__()

        self.mesh = self.build_mesh()
        self.V = self.build_function_space(self.mesh)

        self.setup_problem()

        self.control = self.build_control()
        self.rf = self.build_reduced_functional()
        self.F_torch = fem_operator(self.rf)

    @abstractmethod
    def build_mesh(self):
        pass

    def build_function_space(self, mesh):
        return fd.FunctionSpace(mesh, "CG", 1)

    @abstractmethod
    def setup_problem(self):
        pass

    def build_control(self):
        return fd.Function(self.V, name="control")

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @abstractmethod
    def solve_annotated(self, control: fd.Function) -> fd.Function:
        pass

    def get_dof_coordinates(self):
        #coords = self.V.tabulate_dof_coordinates()
        coords = fd.Function(fd.VectorFunctionSpace(self.V.mesh(),"DG",0)).interpolate(fd.SpatialCoordinate(self.V.mesh())).dat.data # [n_points]
        gdim = self.mesh.geometric_dimension()
        return coords.reshape((-1, gdim))[:, 0]
    
    def build_reduced_functional(self):
        fd.adjoint.continue_annotation()
        try:
            output = self.solve_annotated(self.control)
            rf = ReducedFunctional(output, Control(self.control))
        finally:
            fd.adjoint.stop_annotating()
        return rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_output = False

        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze_output = True

        x = self.preprocess_input(x)
        y = self.F_torch(x)

        if squeeze_output and y.ndim == 2 and y.shape[0] == 1:
            y = y.squeeze(0)

        return y