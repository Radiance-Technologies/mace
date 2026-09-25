###########################################################################################
# Implementation of different loss functions
# Authors: Ilyes Batatia, Gregor Simm, Davy Walker Collins
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Optional

import torch
import torch.distributed as dist
from math import erf, sqrt, pi, exp, log
from abc import ABC, abstractmethod
from enum import Enum
from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch


class LossMode(str, Enum):
    """Available modes for `Loss` calculation."""
    MAE = "mae"
    MSE = "mse"
    HUBER = "huber"
    UNIVERSAL = "universal"


class Loss(torch.nn.Module, ABC):
    """Generic `Loss` module. Defines mode and delta and requires forward."""

    def __init__(
        self,
        mode: LossMode = LossMode.MSE,
        delta: Optional[float] = None,
    ) -> None:
        """
        Create generic `Loss` module for MACE.

        Parameters
        ----------
        mode : LossMode, optional
            Mode for loss calculation, by default LossMode.MSE
        delta : Optional[float], optional
            Delta for huber mode, by default None
        """
        super().__init__()
        self.mode = mode
        self.delta = delta

    @abstractmethod
    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        """
        Forward pass for `Loss`.

        Parameters
        ----------
        ref : Batch
            Batch containing reference information.
        pred : TensorDict
            TensorDict containing prediction information.
        ddp : Optional[bool], optional
            Bool to reduce loss when distributed, by default None

        Returns
        -------
        torch.Tensor
            Loss.
        """
        pass


class SingleLoss(Loss, ABC):
    """Generic `Loss` module for single type of value."""

    def __init__(
        self,
        mode: LossMode = LossMode.MSE,
        delta: Optional[float] = None,
        log_z_delta: Optional[float] = None,
    ) -> None:
        """
        Create generic `SingleLoss` module for single type of value.

        Parameters
        ----------
        mode : LossMode, optional
            Mode for loss calculation, by default LossMode.MSE
        delta : Optional[float], optional
            Delta for huber mode, by default None
        log_z_delta : Optional[float], optional
            log(Z(delta)) for huber nll, by default None
        """
        super().__init__(mode=mode, delta=delta)
        self.log_z_delta = log_z_delta

    def _is_ddp_enabled(self) -> bool:
        """
        Determine if ddp is enabled.

        Returns
        -------
        bool
            True if ddp is enabled, else false.
        """
        return dist.is_initialized() and dist.get_world_size() > 1

    def _reduce_loss(self,
                     raw_loss: torch.Tensor,
                     ddp: Optional[bool] = None) -> torch.Tensor:
        """
        Reduce an element-wise loss tensor.
        
        If ddp is True and distributed is initialized, the function computes:

            loss = (local_sum * world_size) / global_num_elements

        Otherwise, it returns the regular mean.

        Parameters
        ----------
        raw_loss : torch.Tensor
            Loss before reduction.
        ddp : Optional[bool], optional
            Bool to reduce loss when distributed, by default None

        Returns
        -------
        torch.Tensor
            Reduced loss.
        """
        ddp = self._is_ddp_enabled() if ddp is None else ddp
        if ddp and dist.is_initialized():
            world_size = dist.get_world_size()
            n_local = raw_loss.numel()
            loss_sum = raw_loss.sum()
            total_samples = torch.tensor(n_local,
                                         device=raw_loss.device,
                                         dtype=raw_loss.dtype)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
            return loss_sum * world_size / total_samples
        return raw_loss.mean()

    def _assign(self, obj: Batch | TensorDict, key: str) -> torch.Tensor:
        """
        Assign value matching key from obj to variable.

        Parameters
        ----------
        obj : Batch | TensorDict
            Either a ref or pred object.
        key : str
            Key to grab from ref or pred.

        Returns
        -------
        torch.Tensor
            Value matching key from obj.

        Raises
        ------
        ValueError
            Object must have the key.
        """
        if (value := getattr(obj, key, None)) is None:
            raise ValueError(f"{type(obj)} must have {key}.")
        return value

    def _pred_b(self, pred_var: torch.Tensor) -> torch.Tensor:
        """
        Calculate b for laplace nll loss.

        Parameters
        ----------
        pred_var : torch.Tensor
            Variance.

        Returns
        -------
        torch.Tensor
            B.
        """
        return torch.sqrt(pred_var / 2.0)

    def _laplace_nll(self, mae_loss: torch.Tensor,
                     pred_b: torch.Tensor) -> torch.Tensor:
        """
        Calculate laplace nll loss.

        Parameters
        ----------
        mae_loss : torch.Tensor
            Mean Absolute Error.
        pred_b : torch.Tensor
            B.

        Returns
        -------
        torch.Tensor
            Laplace Negative Log Likelihood.
        """
        return mae_loss / pred_b + torch.log(pred_b)

    def _gaussian_nll(self, mse_loss: torch.Tensor,
                      pred_var: torch.Tensor) -> torch.Tensor:
        """
        Calculate gaussian nll loss.

        Parameters
        ----------
        mse_loss : torch.Tensor
            Mean Squared Error.
        pred_var : torch.Tensor
            Variance.

        Returns
        -------
        torch.Tensor
            Gaussian Negative Log Likelihood.
        """
        return 0.5 * (mse_loss / pred_var + torch.log(pred_var))

    def _huber_nll(self, huber_loss: torch.Tensor,
                   pred_std: torch.Tensor) -> torch.Tensor:
        """
        Calculate huber nll loss.

        Parameters
        ----------
        huber_loss : torch.Tensor
            Huber Loss.
        pred_std : torch.Tensor
            Standard Deviation.

        Returns
        -------
        torch.Tensor
            Huber Negative Log Likelihood.

        Raises
        ------
        ValueError
            log(Z(delta)) must exist.
        """
        if self.log_z_delta is None:
            raise ValueError(
                f"log(Z(delta)) must exist when mode is {self.mode}.")
        return huber_loss + torch.log(pred_std) + self.log_z_delta

    def _huber_loss(self, ref_value: torch.Tensor,
                    pred_value: torch.Tensor) -> torch.Tensor:
        """
        Calculate huber loss.

        Parameters
        ----------
        ref_value : torch.Tensor
            Value from reference.
        pred_value : torch.Tensor
            Value from prediction.

        Returns
        -------
        torch.Tensor
            Huber loss.

        Raises
        ------
        ValueError
            Delta must exist.
        """
        if self.delta is None:
            raise ValueError(f"Delta must exist when mode is {self.mode}.")
        return torch.nn.functional.huber_loss(
            ref_value,
            pred_value,
            reduction="none",
            delta=self.delta,
        )

    def _assign_ref_value(self, ref: Batch, key: str) -> torch.Tensor:
        """
        Assign value matching key from ref to variable.

        Parameters
        ----------
        ref : Batch
            Reference.
        key : str
            Key to grab from ref.

        Returns
        -------
        torch.Tensor
            Value matching key from ref.
        """
        return self._assign(ref, key)


class WeightedLoss(SingleLoss, ABC):
    """Generic `Loss` module for weighted loss."""

    def __init__(
        self,
        mode: LossMode = LossMode.MSE,
        delta: Optional[float] = None,
        log_z_delta: Optional[float] = None,
    ) -> None:
        """
        Create generic `WeightedLoss` module for weighted loss.

        Parameters
        ----------
        mode : LossMode, optional
            Mode for loss calculation, by default LossMode.MSE
        delta : Optional[float], optional
            Delta for huber mode, by default None
        log_z_delta : Optional[float], optional
            log(Z(delta)) for huber nll, by default None
        """
        super().__init__(
            mode=mode,
            delta=delta,
            log_z_delta=log_z_delta,
        )

    def _get_weights(self,
                     ref: Batch,
                     key: str,
                     num_atoms: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get weights from reference.

        Parameters
        ----------
        ref : Batch
            Reference.
        key : str
            Weight to grab.
        num_atoms : Optional[torch.Tensor], optional
            Number of Atoms.

        Returns
        -------
        torch.Tensor
            Weight to multiply value by.
        """
        if self.mode == 'huber':
            return torch.tensor([1])

        value_weight = self.assign_value_weight(ref, key, num_atoms)
        if self.mode == 'universal':
            return value_weight

        weight = self._assign_weight(ref, num_atoms)
        return weight * value_weight

    def _assign_weight(
            self,
            ref: Batch,
            num_atoms: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Assign weight from reference to variable.

        Parameters
        ----------
        ref : Batch
            Reference.
        num_atoms : Optional[torch.Tensor], optional
            For when ForcesLoss overrides this method, by default None

        Returns
        -------
        torch.Tensor
            Weight from reference.
        """
        return self._assign(ref, "weight")

    def assign_value_weight(
            self,
            ref: Batch,
            key: str,
            num_atoms: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Assign weight matching key from reference to variable.

        Parameters
        ----------
        ref : Batch
            Reference.
        key : str
            Key for which weight to use.
        num_atoms : Optional[torch.Tensor], optional
            For when ForcesLoss overrides this method, by default None

        Returns
        -------
        torch.Tensor
            Weight matching key from reference.
        """
        return self._assign(ref, f"{key}_weight")


class TotalLoss(SingleLoss, ABC):
    """Generic `TotalLoss` module for loss not split by atom."""

    def __init__(
        self,
        mode: LossMode = LossMode.MSE,
        delta: Optional[float] = None,
        log_z_delta: Optional[float] = None,
    ) -> None:
        """
        Create generic `TotalLoss` module for loss not split by atom.

        Parameters
        ----------
        mode : LossMode, optional
            Mode for loss calculation, by default LossMode.MSE
        delta : Optional[float], optional
            Delta for huber mode, by default None
        log_z_delta : Optional[float], optional
            log(Z(delta)) for huber nll, by default None
        """
        super().__init__(
            mode=mode,
            delta=delta,
            log_z_delta=log_z_delta,
        )

    def _total_mae(self,
                   error: torch.Tensor,
                   pred_var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate MAE or Laplace NLL loss.

        Parameters
        ----------
        error : torch.Tensor
            Error between reference and prediction.
        pred_var : Optional[torch.Tensor], optional
            Variance, by default None

        Returns
        -------
        torch.Tensor
            MAE or Laplace NLL loss.
        """
        loss = torch.abs(error)
        if pred_var is not None:
            pred_b = self._pred_b(pred_var)
            loss = self._laplace_nll(loss, pred_b)
        return loss

    def _total_mse(self,
                   error: torch.Tensor,
                   pred_var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate MSE or Gaussian NLL loss.
        
        Parameters
        ----------
        error : torch.Tensor
            Error between reference and prediction.
        pred_var : Optional[torch.Tensor], optional
            Variance, by default None

        Returns
        -------
        torch.Tensor
            MSE or Gaussian NLL loss.
        """
        loss = torch.square(error)
        if pred_var is not None:
            loss = self._gaussian_nll(loss, pred_var)
        return loss

    def _total_huber(self,
                     ref_value: torch.Tensor,
                     pred_value: torch.Tensor,
                     pred_var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate Huber or Huber NLL loss.

        Parameters
        ----------
        ref_value : torch.Tensor
            Value from reference.
        pred_value : torch.Tensor
            Value from prediction.
        pred_var : Optional[torch.Tensor], optional
            Variance, by default None

        Returns
        -------
        torch.Tensor
            Huber or Huber NLL loss.
        """
        if pred_var is not None:
            pred_std = torch.sqrt(pred_var)
            inv_pred_std = 1.0 / pred_std
            ref_value *= inv_pred_std
            pred_value *= inv_pred_std
        loss = self._huber_loss(ref_value, pred_value)
        if pred_var is not None:
            loss = self._huber_nll(loss, pred_std)
        return loss

    def _total_loss(self,
                    ref: Batch,
                    pred: TensorDict,
                    ddp: Optional[bool] = None,
                    key: str = 'stress',
                    weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate loss.

        Parameters
        ----------
        ref : Batch
            Batch containing reference information.
        pred : TensorDict
            TensorDict containing prediction information.
        ddp : Optional[bool], optional
            Bool to reduce loss when distributed, by default None
        key : str, optional
            Value to take loss of, by default 'stress'
        weights : Optional[torch.Tensor], optional
            Weights to multiply value by, by default None

        Returns
        -------
        torch.Tensor
            Loss.

        Raises
        ------
        ValueError
            Mode must be in LossMode.
        """
        pred_var: Optional[torch.Tensor] = getattr(pred, f"{key}_var", None)
        ref_value = self._assign_ref_value(ref, key)
        pred_value = self._assign(pred, key)
        error = ref_value - pred_value
        match self.mode:
            case 'huber':
                loss = self._total_huber(ref_value, pred_value, pred_var)
            case 'universal':
                if weights is not None:
                    ref_value *= weights
                    pred_value *= weights
                loss = self._total_huber(ref_value, pred_value, pred_var)
            case 'mae':
                loss = self._total_mae(error, pred_var)
                if weights is not None:
                    loss *= weights
            case 'mse':
                loss = self._total_mse(error, pred_var)
                if weights is not None:
                    loss *= weights
            case _:
                raise ValueError(f"Mode {self.mode} must be in LossMode.")
        return self._reduce_loss(loss, ddp)


class WeightedTotalLoss(TotalLoss, WeightedLoss, ABC):
    """Generic `WeightedTotalLoss` module for weighted total loss."""

    def __init__(
        self,
        mode: LossMode = LossMode.MSE,
        delta: Optional[float] = None,
        log_z_delta: Optional[float] = None,
    ) -> None:
        """
        Create generic `WeightedTotalLoss` module for weighted total loss.

        Parameters
        ----------
        mode : LossMode, optional
            Mode for loss calculation, by default LossMode.MSE
        delta : Optional[float], optional
            Delta for huber mode, by default None
        log_z_delta : Optional[float], optional
            log(Z(delta)) for huber nll, by default None
        """
        super().__init__(
            mode=mode,
            delta=delta,
            log_z_delta=log_z_delta,
        )

    def _weighted_total_loss(self,
                             ref: Batch,
                             pred: TensorDict,
                             ddp: Optional[bool] = None,
                             key: str = 'forces') -> torch.Tensor:
        """
        Calculate loss.

        Parameters
        ----------
        ref : Batch
            Batch containing reference information.
        pred : TensorDict
            TensorDict containing prediction information.
        ddp : Optional[bool], optional
            Bool to reduce loss when distributed, by default None
        key : str, optional
            Value to take loss of, by default 'stress'

        Returns
        -------
        torch.Tensor
            Loss.
        """
        weights = self._get_weights(ref, key)
        return self._total_loss(ref, pred, ddp, key, weights)


class LossPerAtom(SingleLoss, ABC):
    """Generic `LossPerAtom` module for loss split by atom."""

    def __init__(
        self,
        mode: LossMode = LossMode.MSE,
        delta: Optional[float] = None,
        log_z_delta: Optional[float] = None,
    ) -> None:
        """
        Create generic `LossPerAtom` module for loss split by atom.

        Parameters
        ----------
        mode : LossMode, optional
            Mode for loss calculation, by default LossMode.MSE
        delta : Optional[float], optional
            Delta for huber mode, by default None
        log_z_delta : Optional[float], optional
            log(Z(delta)) for huber nll, by default None
        """
        super().__init__(
            mode=mode,
            delta=delta,
            log_z_delta=log_z_delta,
        )

    def _mae_per_atom(self,
                      num_atoms: torch.Tensor,
                      error_per_atom: torch.Tensor,
                      pred_var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate MAE or Laplace NLL loss.

        Parameters
        ----------
        num_atoms : torch.Tensor
            Number of atoms.
        error_per_atom : torch.Tensor
            Error between reference and prediction.
        pred_var : Optional[torch.Tensor], optional
            Variance, by default None

        Returns
        -------
        torch.Tensor
            MAE or Laplace NLL loss.
        """
        loss = torch.abs(error_per_atom)
        if pred_var is not None:
            pred_b_per_atom = self._pred_b(pred_var) / num_atoms
            loss = self._laplace_nll(loss, pred_b_per_atom)
        return loss

    def _mse_per_atom(self,
                      num_atoms: torch.Tensor,
                      error_per_atom: torch.Tensor,
                      pred_var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate MSE or Gaussian NLL loss.
        
        Parameters
        ----------
        num_atoms : torch.Tensor
            Number of atoms.
        error_per_atom : torch.Tensor
            Error between reference and prediction.
        pred_var : Optional[torch.Tensor], optional
            Variance, by default None

        Returns
        -------
        torch.Tensor
            MSE or Gaussian NLL loss.
        """
        loss = torch.square(error_per_atom)
        if pred_var is not None:
            pred_var_per_atom = pred_var / (num_atoms**2)
            loss = self._gaussian_nll(loss, pred_var_per_atom)
        return loss

    def _huber_per_atom(
            self,
            num_atoms: torch.Tensor,
            ref_value_per_atom: torch.Tensor,
            pred_value_per_atom: torch.Tensor,
            pred_var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate Huber or Huber NLL loss.

        Parameters
        ----------
        num_atoms : torch.Tensor
            Number of atoms.
        ref_value_per_atom : torch.Tensor
            Value from reference.
        pred_value_per_atom : torch.Tensor
            Value from prediction.
        pred_var : Optional[torch.Tensor], optional
            Variance, by default None

        Returns
        -------
        torch.Tensor
            Huber or Huber NLL loss.
        """
        if pred_var is not None:
            pred_std_per_atom = torch.sqrt(pred_var) / num_atoms
            inv_pred_std_per_atom = 1.0 / pred_std_per_atom
            ref_value_per_atom *= inv_pred_std_per_atom
            pred_value_per_atom *= inv_pred_std_per_atom
        loss = self._huber_loss(ref_value_per_atom, pred_value_per_atom)
        if pred_var is not None:
            loss = self._huber_nll(loss, pred_std_per_atom)
        return loss

    def _loss_per_atom(self,
                       ref: Batch,
                       pred: TensorDict,
                       ddp: Optional[bool] = None,
                       key: str = 'energy',
                       weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate loss.

        Parameters
        ----------
        ref : Batch
            Batch containing reference information.
        pred : TensorDict
            TensorDict containing prediction information.
        ddp : Optional[bool], optional
            Bool to reduce loss when distributed, by default None
        key : str, optional
            Value to take loss of, by default 'energy'
        weights : Optional[torch.Tensor], optional
            Weights to multiply value by, by default None

        Returns
        -------
        torch.Tensor
            Loss.

        Raises
        ------
        ValueError
            Mode must be in LossMode.
        """
        ptr = self._assign(ref, "ptr")
        num_atoms = self._assign_num_atoms(ptr)
        pred_var: Optional[torch.Tensor] = getattr(pred, f"{key}_var", None)
        ref_value_per_atom = self._assign_ref_value(ref, key) / num_atoms
        pred_value_per_atom = self._assign(pred, key) / num_atoms
        error_per_atom = ref_value_per_atom - pred_value_per_atom

        match self.mode:
            case 'huber':
                loss_per_atom = self._huber_per_atom(num_atoms,
                                                     ref_value_per_atom,
                                                     pred_value_per_atom,
                                                     pred_var)
            case 'universal':
                if weights is not None:
                    ref_value_per_atom *= weights
                    pred_value_per_atom *= weights
                loss_per_atom = self._huber_per_atom(num_atoms,
                                                     ref_value_per_atom,
                                                     pred_value_per_atom,
                                                     pred_var)
            case 'mae':
                loss_per_atom = self._mae_per_atom(num_atoms, error_per_atom,
                                                   pred_var)
                if weights is not None:
                    loss_per_atom *= weights
            case 'mse':
                loss_per_atom = self._mse_per_atom(num_atoms, error_per_atom,
                                                   pred_var)
                if weights is not None:
                    loss_per_atom *= weights
            case _:
                raise ValueError(f"Mode {self.mode} must be in LossMode.")
        return self._reduce_loss(loss_per_atom, ddp)

    def _assign_num_atoms(self, ptr: torch.Tensor) -> torch.Tensor:
        """
        Calculate number of atoms.

        Parameters
        ----------
        ptr : torch.Tensor
            Pointer.

        Returns
        -------
        torch.Tensor
            Number of atoms.
        """
        return ptr[1:] - ptr[:-1]


class WeightedLossPerAtom(LossPerAtom, WeightedLoss, ABC):

    def __init__(
        self,
        mode: LossMode = LossMode.MSE,
        delta: Optional[float] = None,
        log_z_delta: Optional[float] = None,
    ) -> None:
        """
        Create generic `WeightedLossPerAtom` module for split weighted loss.

        Parameters
        ----------
        mode : LossMode, optional
            Mode for loss calculation, by default LossMode.MSE
        delta : Optional[float], optional
            Delta for huber mode, by default None
        log_z_delta : Optional[float], optional
            log(Z(delta)) for huber nll, by default None
        """
        super().__init__(
            mode=mode,
            delta=delta,
            log_z_delta=log_z_delta,
        )

    def _weighted_loss_per_atom(self,
                                ref: Batch,
                                pred: TensorDict,
                                ddp: Optional[bool] = None,
                                key: str = 'energy') -> torch.Tensor:
        """
        Calculate loss.

        Parameters
        ----------
        ref : Batch
            Batch containing reference information.
        pred : TensorDict
            TensorDict containing prediction information.
        ddp : Optional[bool], optional
            Bool to reduce loss when distributed, by default None
        key : str, optional
            Value to take loss of, by default 'energy'

        Returns
        -------
        torch.Tensor
            Loss.
        """
        weights = self._get_weights(ref, key)
        return self._loss_per_atom(ref, pred, ddp, key, weights)


class EnergyLoss(WeightedLossPerAtom):
    """`EnergyLoss` module for loss from energy."""
    energy_weight: torch.Tensor

    def __init__(
        self,
        energy_weight: float = 1.0,
        mode: LossMode = LossMode.MSE,
        delta: Optional[float] = None,
        log_z_delta: Optional[float] = None,
    ) -> None:
        """
        Create `EnergyLoss` module for loss from energy.

        Parameters
        ----------
        energy_weight : float, optional
            Weight for energy loss, by default 1.0
        mode : LossMode, optional
            Mode for loss calculation, by default LossMode.MSE
        delta : Optional[float], optional
            Delta for huber mode, by default None
        log_z_delta : Optional[float], optional
            log(Z(delta)) for huber nll, by default None
        """
        super().__init__(
            mode=mode,
            delta=delta,
            log_z_delta=log_z_delta,
        )
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        """
        Forward pass for `EnergyLoss`.
        
        Parameters
        ----------
        ref : Batch
            Batch containing reference information.
        pred : TensorDict
            TensorDict containing prediction information.
        ddp : Optional[bool], optional
            Bool to reduce loss when distributed, by default None

        Returns
        -------
        torch.Tensor
            Loss.
        """
        loss = self._weighted_loss_per_atom(ref, pred, ddp, 'energy')
        return self.energy_weight * loss

    def __repr__(self):
        """
        Create string representation.

        Returns
        -------
        str
            String representation.
        """
        return (f"{self.__class__.__name__}(energy_weight="
                f"{self.energy_weight.item():.3f})")


class VirialsLoss(WeightedLossPerAtom):
    """`VirialsLoss` module for loss from virials."""
    virials_weight: torch.Tensor

    def __init__(
        self,
        virials_weight: float = 1.0,
        mode: LossMode = LossMode.MSE,
        delta: Optional[float] = None,
        log_z_delta: Optional[float] = None,
    ) -> None:
        """
        Create `VirialsLoss` module for loss from virials.

        Parameters
        ----------
        virials_weight : float, optional
            Weight for virials loss, by default 1.0
        mode : LossMode, optional
            Mode for loss calculation, by default LossMode.MSE
        delta : Optional[float], optional
            Delta for huber mode, by default None
        log_z_delta : Optional[float], optional
            log(Z(delta)) for huber nll, by default None
        """
        super().__init__(
            mode=mode,
            delta=delta,
            log_z_delta=log_z_delta,
        )
        self.register_buffer(
            "virials_weight",
            torch.tensor(virials_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        """
        Forward pass for `VirialsLoss`.
        
        Parameters
        ----------
        ref : Batch
            Batch containing reference information.
        pred : TensorDict
            TensorDict containing prediction information.
        ddp : Optional[bool], optional
            Bool to reduce loss when distributed, by default None

        Returns
        -------
        torch.Tensor
            Loss.
        """
        loss = self._weighted_loss_per_atom(ref, pred, ddp, 'virials')
        return self.virials_weight * loss

    def _assign_num_atoms(self, ptr: torch.Tensor) -> torch.Tensor:
        """
        Calculate number of atoms.

        Parameters
        ----------
        ptr : torch.Tensor
            Pointer.

        Returns
        -------
        torch.Tensor
            Number of atoms.
        """
        return (ptr[1:] - ptr[:-1]).view(-1, 1, 1)

    def _assign_weight(
            self,
            ref: Batch,
            num_atoms: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Assign weight from reference to variable.

        Parameters
        ----------
        ref : Batch
            Reference.
        num_atoms : Optional[torch.Tensor], optional
            For when ForcesLoss overrides this method, by default None

        Returns
        -------
        torch.Tensor
            Weight from reference.
        """
        return self._assign(ref, "weight").view(-1, 1, 1)

    def assign_value_weight(
            self,
            ref: Batch,
            key: str,
            num_atoms: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Assign weight matching key from reference to variable.

        Parameters
        ----------
        ref : Batch
            Reference.
        key : str
            Key for which weight to use.
        num_atoms : Optional[torch.Tensor], optional
            For when ForcesLoss overrides this method, by default None

        Returns
        -------
        torch.Tensor
            Weight matching key from reference.
        """
        return self._assign(ref, f"{key}_weight").view(-1, 1, 1)

    def __repr__(self):
        """
        Create string representation.

        Returns
        -------
        str
            String representation.
        """
        return (f"{self.__class__.__name__}(virials_weight="
                f"{self.virials_weight.item():.3f})")


class PolarizabilityLoss(LossPerAtom):
    """`PolarizabilityLoss` module for loss from polarizability."""
    polarizability_weight: torch.Tensor

    def __init__(
        self,
        polarizability_weight: float = 1.0,
        mode: LossMode = LossMode.MSE,
        delta: Optional[float] = None,
        log_z_delta: Optional[float] = None,
    ) -> None:
        """
        Create `PolarizabilityLoss` module for loss from polarizability.

        Parameters
        ----------
        polarizability_weight : float, optional
            Weight for polarizability loss, by default 1.0
        mode : LossMode, optional
            Mode for loss calculation, by default LossMode.MSE
        delta : Optional[float], optional
            Delta for huber mode, by default None
        log_z_delta : Optional[float], optional
            log(Z(delta)) for huber nll, by default None
        """
        super().__init__(
            mode=mode,
            delta=delta,
            log_z_delta=log_z_delta,
        )
        self.register_buffer(
            "polarizability_weight",
            torch.tensor(polarizability_weight,
                         dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        """
        Forward pass for `PolarizabilityLoss`.
        
        Parameters
        ----------
        ref : Batch
            Batch containing reference information.
        pred : TensorDict
            TensorDict containing prediction information.
        ddp : Optional[bool], optional
            Bool to reduce loss when distributed, by default None

        Returns
        -------
        torch.Tensor
            Loss.
        """
        loss = self._loss_per_atom(ref, pred, ddp, 'polarizability')
        return self.polarizability_weight * loss

    def _assign_num_atoms(self, ptr: torch.Tensor) -> torch.Tensor:
        """
        Calculate number of atoms.

        Parameters
        ----------
        ptr : torch.Tensor
            Pointer.

        Returns
        -------
        torch.Tensor
            Number of atoms.
        """
        return (ptr[1:] - ptr[:-1]).view(-1, 1, 1)

    def _assign_ref_value(self, ref: Batch, key: str) -> torch.Tensor:
        """
        Assign value matching key from ref to variable.

        Parameters
        ----------
        ref : Batch
            Reference.
        key : str
            Key to grab from ref.

        Returns
        -------
        torch.Tensor
            Value matching key from ref.
        """
        return self._assign(ref, key).view(-1, 3, 3)

    def __repr__(self):
        """
        Create string representation.

        Returns
        -------
        str
            String representation.
        """
        return (f"{self.__class__.__name__}(polarizability_weight="
                f"{self.polarizability_weight.item():.3f})")


class DipoleLoss(LossPerAtom):
    """`DipoleLoss` module for loss from dipoles."""
    dipole_weight: torch.Tensor

    def __init__(
        self,
        dipole_weight: float = 1.0,
        mode: LossMode = LossMode.MSE,
        delta: Optional[float] = None,
        log_z_delta: Optional[float] = None,
    ) -> None:
        """
        Create `DipoleLoss` module for loss from dipoles.

        Parameters
        ----------
        dipole_weight : float, optional
            Weight for dipoles loss, by default 1.0
        mode : LossMode, optional
            Mode for loss calculation, by default LossMode.MSE
        delta : Optional[float], optional
            Delta for huber mode, by default None
        log_z_delta : Optional[float], optional
            log(Z(delta)) for huber nll, by default None
        """
        super().__init__(
            mode=mode,
            delta=delta,
            log_z_delta=log_z_delta,
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        """
        Forward pass for `DipoleLoss`.
        
        Parameters
        ----------
        ref : Batch
            Batch containing reference information.
        pred : TensorDict
            TensorDict containing prediction information.
        ddp : Optional[bool], optional
            Bool to reduce loss when distributed, by default None

        Returns
        -------
        torch.Tensor
            Loss.
        """
        loss = self._loss_per_atom(ref, pred, ddp, 'dipole')
        return self.dipole_weight * loss

    def _assign_num_atoms(self, ptr: torch.Tensor) -> torch.Tensor:
        """
        Calculate number of atoms.

        Parameters
        ----------
        ptr : torch.Tensor
            Pointer.

        Returns
        -------
        torch.Tensor
            Number of atoms.
        """
        return (ptr[1:] - ptr[:-1]).unsqueeze(-1)

    def __repr__(self):
        """
        Create string representation.

        Returns
        -------
        str
            String representation.
        """
        return (f"{self.__class__.__name__}(dipole_weight="
                f"{self.dipole_weight.item():.3f})")


class StressLoss(WeightedTotalLoss):
    """`StressLoss` module for loss from stress."""
    stress_weight: torch.Tensor

    def __init__(
        self,
        stress_weight: float = 1.0,
        mode: LossMode = LossMode.MSE,
        delta: Optional[float] = None,
        log_z_delta: Optional[float] = None,
    ) -> None:
        """
        Create `StressLoss` module for loss from stress.

        Parameters
        ----------
        stress_weight : float, optional
            Weight for stress loss, by default 1.0
        mode : LossMode, optional
            Mode for loss calculation, by default LossMode.MSE
        delta : Optional[float], optional
            Delta for huber mode, by default None
        log_z_delta : Optional[float], optional
            log(Z(delta)) for huber nll, by default None
        """
        super().__init__(
            mode=mode,
            delta=delta,
            log_z_delta=log_z_delta,
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        """
        Forward pass for `StressLoss`.
        
        Parameters
        ----------
        ref : Batch
            Batch containing reference information.
        pred : TensorDict
            TensorDict containing prediction information.
        ddp : Optional[bool], optional
            Bool to reduce loss when distributed, by default None

        Returns
        -------
        torch.Tensor
            Loss.
        """
        loss = self._weighted_total_loss(ref, pred, ddp, 'stress')
        return self.stress_weight * loss

    def _assign_weight(
            self,
            ref: Batch,
            num_atoms: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Assign weight from reference to variable.

        Parameters
        ----------
        ref : Batch
            Reference.
        num_atoms : Optional[torch.Tensor], optional
            For when ForcesLoss overrides this method, by default None

        Returns
        -------
        torch.Tensor
            Weight from reference.
        """
        return self._assign(ref, "weight").view(-1, 1, 1)

    def assign_value_weight(
            self,
            ref: Batch,
            key: str,
            num_atoms: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Assign weight matching key from reference to variable.

        Parameters
        ----------
        ref : Batch
            Reference.
        key : str
            Key for which weight to use.
        num_atoms : Optional[torch.Tensor], optional
            For when ForcesLoss overrides this method, by default None

        Returns
        -------
        torch.Tensor
            Weight matching key from reference.
        """
        return self._assign(ref, f"{key}_weight").view(-1, 1, 1)

    def __repr__(self):
        """
        Create string representation.

        Returns
        -------
        str
            String representation.
        """
        return (
            f"{self.__class__.__name__}(stress_weight={self.stress_weight.item():.3f})"
        )


class ForcesLoss(WeightedTotalLoss):
    """`ForcesLoss` module for loss from forces."""
    forces_weight: torch.Tensor

    def __init__(
        self,
        forces_weight: float = 1.0,
        mode: LossMode = LossMode.MSE,
        delta: Optional[float] = None,
        log_z_delta: Optional[float] = None,
    ) -> None:
        """
        Create `ForcesLoss` module for loss from forces.

        Parameters
        ----------
        forces_weight : float, optional
            Weight for forces loss, by default 1.0
        mode : LossMode, optional
            Mode for loss calculation, by default LossMode.MSE
        delta : Optional[float], optional
            Delta for huber mode, by default None
        log_z_delta : Optional[float], optional
            log(Z(delta)) for huber nll, by default None
        """
        super().__init__(
            mode=mode,
            delta=delta,
            log_z_delta=log_z_delta,
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None):
        """
            Forward pass for `ForcesLoss`.
            
            Parameters
            ----------
            ref : Batch
                Batch containing reference information.
            pred : TensorDict
                TensorDict containing prediction information.
            ddp : Optional[bool], optional
                Bool to reduce loss when distributed, by default None
    
            Returns
            -------
            torch.Tensor
                Loss.
            """
        ptr = self._assign(ref, "ptr")
        num_atoms = self._assign_num_atoms(ptr)
        weights = self._get_weights(ref, 'forces', num_atoms)
        pred_var: Optional[torch.Tensor] = getattr(pred, f"forces_var", None)
        ref_value = self._assign_ref_value(ref, 'forces')
        pred_value = self._assign(pred, 'forces')
        error = ref_value - pred_value
        match self.mode:
            case 'huber':
                loss = self._total_huber(ref_value, pred_value, pred_var)
            case 'universal':
                if weights is not None:
                    ref_value *= weights
                    pred_value *= weights
                loss = self._conditional_huber(ref_value, pred_value, pred_var)
            case 'mae':
                loss = self._total_mae(error, pred_var)
            case 'mse':
                loss = self._total_mse(error, pred_var)
                if weights is not None:
                    loss *= weights
            case _:
                raise ValueError(
                    f"Mode {self.mode} must match 'mae', 'mse', 'huber', or "
                    "'universal'.")
        return self._reduce_loss(loss, ddp)

    def _total_mae(self,
                   error: torch.Tensor,
                   pred_var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate MAE or Laplace NLL loss.

        Parameters
        ----------
        error : torch.Tensor
            Error between reference and prediction.
        pred_var : Optional[torch.Tensor], optional
            Variance, by default None

        Returns
        -------
        torch.Tensor
            MAE or Laplace NLL loss.
        """
        loss = torch.linalg.vector_norm(error, ord=2, dim=-1)
        if pred_var is not None:
            pred_b = self._pred_b(pred_var)
            loss = self._laplace_nll(loss, pred_b)
        return loss

    def _conditional_huber(
            self,
            ref_value: torch.Tensor,
            pred_value: torch.Tensor,
            pred_var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate conditional Huber or Huber NLL loss.

        Parameters
        ----------
        ref_value : torch.Tensor
            Value from reference.
        pred_value : torch.Tensor
            Value from prediction.
        pred_var : Optional[torch.Tensor], optional
            Variance, by default None

        Returns
        -------
        torch.Tensor
            Conditional Huber or Huber NLL loss.
        """
        if pred_var is not None:
            pred_std = torch.sqrt(pred_var)
            inv_pred_std = 1.0 / pred_std
            ref_value *= inv_pred_std
            pred_value *= inv_pred_std
        loss = self._conditional_huber_loss(ref_value, pred_value)
        if pred_var is not None:
            loss = self._huber_nll(loss, pred_std)
        return loss

    def _conditional_huber_loss(self, ref_value: torch.Tensor,
                                pred_value: torch.Tensor) -> torch.Tensor:
        """
        Calculate conditional huber loss.

        Parameters
        ----------
        ref_value : torch.Tensor
            Value from reference.
        pred_value : torch.Tensor
            Value from prediction.

        Returns
        -------
        torch.Tensor
            Conditional Huber loss.

        Raises
        ------
        ValueError
            Delta must exist.
        """
        if self.delta is None:
            raise ValueError(f"Delta must exist when mode is {self.mode}.")
        factors = [self.delta * x for x in [1.0, 0.7, 0.4, 0.1]]
        norm_value = torch.norm(ref_value, dim=-1)
        c1 = norm_value < 100
        c2 = (norm_value >= 100) & (norm_value < 200)
        c3 = (norm_value >= 200) & (norm_value < 300)
        c4 = ~(c1 | c2 | c3)
        huber = torch.zeros_like(pred_value)
        huber[c1] = torch.nn.functional.huber_loss(ref_value[c1],
                                                   pred_value[c1],
                                                   reduction="none",
                                                   delta=factors[0])
        huber[c2] = torch.nn.functional.huber_loss(ref_value[c2],
                                                   pred_value[c2],
                                                   reduction="none",
                                                   delta=factors[1])
        huber[c3] = torch.nn.functional.huber_loss(ref_value[c3],
                                                   pred_value[c3],
                                                   reduction="none",
                                                   delta=factors[2])
        huber[c4] = torch.nn.functional.huber_loss(ref_value[c4],
                                                   pred_value[c4],
                                                   reduction="none",
                                                   delta=factors[3])
        return huber

    def _assign_num_atoms(self, ptr: torch.Tensor) -> torch.Tensor:
        """
        Calculate number of atoms.

        Parameters
        ----------
        ptr : torch.Tensor
            Pointer.

        Returns
        -------
        torch.Tensor
            Number of atoms.
        """
        return ptr[1:] - ptr[:-1]

    def _assign_weight(
            self,
            ref: Batch,
            num_atoms: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Assign weight from reference to variable.

        Parameters
        ----------
        ref : Batch
            Reference.
        num_atoms : Optional[torch.Tensor], optional
            Number of atoms, by default None

        Returns
        -------
        torch.Tensor
            Weight from reference.

        Raises
        ------
        ValueError
            Number of atoms must not be None.
        """
        weight = self._assign(ref, "weight")
        if num_atoms is not None:
            return torch.repeat_interleave(weight, num_atoms).unsqueeze(-1)
        else:
            raise ValueError("Number of atoms must not be None.")

    def assign_value_weight(
            self,
            ref: Batch,
            key: str,
            num_atoms: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Assign weight matching key from reference to variable.

        Parameters
        ----------
        ref : Batch
            Reference.
        key : str
            Key for which weight to use.
        num_atoms : Optional[torch.Tensor], optional
            Number of atoms, by default None

        Returns
        -------
        torch.Tensor
            Weight matching key from reference.

        Raises
        ------
        ValueError
            Number of atoms must not be None.
        """
        value_weight = self._assign(ref, f"{key}_weight")
        if num_atoms is not None:
            return torch.repeat_interleave(value_weight,
                                           num_atoms).unsqueeze(-1)
        else:
            raise ValueError("Number of atoms must not be None.")

    def __repr__(self):
        """
        Create string representation.

        Returns
        -------
        str
            String representation.
        """
        return (
            f"{self.__class__.__name__}(forces_weight={self.forces_weight.item():.3f})"
        )


class CombinedLoss(Loss):
    """`CombinedLoss` module for multiple types of value."""
    _LOSS_MAP = {
        "energy_weight": EnergyLoss,
        "virials_weight": VirialsLoss,
        "polarizability_weight": PolarizabilityLoss,
        "dipole_weight": DipoleLoss,
        "stress_weight": StressLoss,
        "forces_weight": ForcesLoss,
    }

    def __init__(self,
                 mode: LossMode = LossMode.MSE,
                 delta: Optional[float] = None,
                 **weights: float) -> None:
        """
        Create `CombinedLoss` module for multiple types of values.

        Parameters
        ----------
        mode : LossMode, optional
            Mode for loss calculation, by default LossMode.MSE
        delta : Optional[float], optional
            Delta for huber mode, by default None
        weights : dict[str, float]
            Dictionary of passed weights for types of values.
        """
        super().__init__(mode=mode, delta=delta)
        self._log_z_delta: Optional[float] = None
        self.losses: torch.nn.ModuleList = torch.nn.ModuleList([
            cls(weight, mode, delta, self.log_z_delta)
            for key, cls in self._LOSS_MAP.items()
            if (weight := weights.get(key)) is not None
        ])

    @property
    def log_z_delta(self) -> Optional[float]:
        """
        Getter for log(Z(delta)).

        Returns
        -------
        Optional[float]
            Log of Z(delta) if delta is not None, else None.
        """
        if self._log_z_delta is not None:
            return self._log_z_delta
        if self.delta is None:
            return None
        cdf = 0.5 * (1.0 + erf(self.delta / sqrt(2)))
        pdf = (1.0 / sqrt(2 * pi)) * exp(-0.5 * self.delta**2)
        return log(sqrt(2 * pi) * cdf + self.delta * pdf)

    @log_z_delta.setter
    def log_z_delta(self, log_z_delta: Optional[float]) -> None:
        """
        Setter for log(Z(delta)).

        Parameters
        ----------
        log_z_delta : Optional[float]
            Log of Z(delta).
        """
        self._log_z_delta = log_z_delta

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None):
        """
        Forward pass for `CombinedLoss`, summing each loss.

        Parameters
        ----------
        ref : Batch
            Batch containing reference information.
        pred : TensorDict
            TensorDict containing prediction information.
        ddp : Optional[bool], optional
            Bool to reduce loss when distributed, by default None

        Returns
        -------
        torch.Tensor
            Summed Loss.
        """
        return sum(loss(ref, pred, ddp) for loss in self.losses)

    def __repr__(self):
        """
        Create string representation.

        Returns
        -------
        str
            String representation.
        """
        lines = [f"mode={self.mode!r}"]

        loss_lines = []
        for idx, loss in enumerate(self.losses):
            loss_repr = repr(loss)
            indented_loss = torch.nn.modules.module._addindent(loss_repr, 2)
            loss_lines.append(f"({idx}): {indented_loss}")

        if loss_lines:
            losses_formatted = ",\n".join(loss_lines)
            lines.append(
                f"losses=[\n{torch.nn.modules.module._addindent(losses_formatted, 2)}\n]"
            )

        main_str = self.__class__.__name__ + "(\n"
        main_str += torch.nn.modules.module._addindent("\n".join(lines), 2)
        main_str += "\n)"

        return main_str
