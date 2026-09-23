###########################################################################################
# Implementation of different loss functions
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Optional

import torch
import torch.distributed as dist

from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch


# ------------------------------------------------------------------------------
# Helper function for loss reduction that handles DDP correction
# ------------------------------------------------------------------------------
def is_ddp_enabled():
    return dist.is_initialized() and dist.get_world_size() > 1


def reduce_loss(raw_loss: torch.Tensor,
                ddp: Optional[bool] = None) -> torch.Tensor:
    """
    Reduces an element-wise loss tensor.

    If ddp is True and distributed is initialized, the function computes:

        loss = (local_sum * world_size) / global_num_elements

    Otherwise, it returns the regular mean.
    """
    ddp = is_ddp_enabled() if ddp is None else ddp
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


# ------------------------------------------------------------------------------
# Energy Loss Functions
# ------------------------------------------------------------------------------


def mean_squared_error_energy(ref: Batch,
                              pred: TensorDict,
                              ddp: Optional[bool] = None) -> torch.Tensor:
    raw_loss = torch.square(ref["energy"] - pred["energy"])
    return reduce_loss(raw_loss, ddp)


def gaussian_nll_energy(
        ref: Batch,
        pred: TensorDict,
        ddp: Optional[bool] = None) -> tuple[torch.Tensor, torch.Tensor]:
    log_var = torch.log(pred["energy_var"])

    raw_mse = torch.square(ref["energy"] - pred["energy"])
    raw_nll = 0.5 * (raw_mse / pred["energy_var"] + log_var)

    return reduce_loss(raw_mse, ddp), reduce_loss(raw_nll, ddp)


def weighted_mean_squared_error_energy(
        ref: Batch,
        pred: TensorDict,
        ddp: Optional[bool] = None) -> torch.Tensor:
    # Calculate per-graph number of atoms.
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # shape: [n_graphs]
    raw_loss = (ref.weight * ref.energy_weight * torch.square(
        (ref["energy"] - pred["energy"]) / num_atoms))
    return reduce_loss(raw_loss, ddp)


def weighted_gaussian_nll_energy(
        ref: Batch,
        pred: TensorDict,
        ddp: Optional[bool] = None) -> tuple[torch.Tensor, torch.Tensor]:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # shape: [n_graphs]
    weights = ref.weight * ref.energy_weight

    per_atom_energy_var = pred["energy_var"] / (num_atoms**2)
    log_var = torch.log(per_atom_energy_var)

    raw_mse = torch.square((ref["energy"] - pred["energy"]) / num_atoms)
    raw_nll = 0.5 * (raw_mse / per_atom_energy_var + log_var)

    return reduce_loss(weights * raw_mse,
                       ddp), reduce_loss(weights * raw_nll, ddp)


def weighted_mean_absolute_error_energy(
        ref: Batch,
        pred: TensorDict,
        ddp: Optional[bool] = None) -> torch.Tensor:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    raw_loss = (ref.weight * ref.energy_weight * torch.abs(
        (ref["energy"] - pred["energy"]) / num_atoms))
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Stress and Virials Loss Functions
# ------------------------------------------------------------------------------


def weighted_mean_squared_stress(ref: Batch,
                                 pred: TensorDict,
                                 ddp: Optional[bool] = None) -> torch.Tensor:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
    raw_loss = (configs_weight * configs_stress_weight *
                torch.square(ref["stress"] - pred["stress"]))
    return reduce_loss(raw_loss, ddp)


def weighted_gaussian_nll_stress(
        ref: Batch,
        pred: TensorDict,
        ddp: Optional[bool] = None) -> tuple[torch.Tensor, torch.Tensor]:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
    weights = configs_weight * configs_stress_weight

    log_var = torch.log(pred["stress_var"])

    raw_mse = torch.square(ref["stress"] - pred["stress"])
    raw_nll = 0.5 * (raw_mse / pred["stress_var"] + log_var)

    return reduce_loss(weights * raw_mse,
                       ddp), reduce_loss(weights * raw_nll, ddp)


def weighted_mean_squared_virials(ref: Batch,
                                  pred: TensorDict,
                                  ddp: Optional[bool] = None) -> torch.Tensor:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_virials_weight = ref.virials_weight.view(-1, 1, 1)
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)
    raw_loss = (configs_weight * configs_virials_weight * torch.square(
        (ref["virials"] - pred["virials"]) / num_atoms))
    return reduce_loss(raw_loss, ddp)


def weighted_gaussian_nll_virials(
        ref: Batch,
        pred: TensorDict,
        ddp: Optional[bool] = None) -> tuple[torch.Tensor, torch.Tensor]:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_virials_weight = ref.virials_weight.view(-1, 1, 1)
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)
    weights = configs_weight * configs_virials_weight

    per_atom_virial_var = pred["virials_var"] / (num_atoms**2)
    log_var = torch.log(per_atom_virial_var)

    raw_mse = torch.square((ref["virials"] - pred["virials"]) / num_atoms)
    raw_nll = 0.5 * (raw_mse / per_atom_virial_var + log_var)

    return reduce_loss(weights * raw_mse,
                       ddp), reduce_loss(weights * raw_nll, ddp)


# ------------------------------------------------------------------------------
# Forces Loss Functions
# ------------------------------------------------------------------------------


def mean_squared_error_forces(ref: Batch,
                              pred: TensorDict,
                              ddp: Optional[bool] = None) -> torch.Tensor:
    # Repeat per-graph weights to per-atom level.
    configs_weight = torch.repeat_interleave(ref.weight, ref.ptr[1:] -
                                             ref.ptr[:-1]).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
    raw_loss = (configs_weight * configs_forces_weight *
                torch.square(ref["forces"] - pred["forces"]))
    return reduce_loss(raw_loss, ddp)


def gaussian_nll_forces(
        ref: Batch,
        pred: TensorDict,
        ddp: Optional[bool] = None) -> tuple[torch.Tensor, torch.Tensor]:
    # Repeat per-graph weights to per-atom level
    configs_weight = torch.repeat_interleave(ref.weight, ref.ptr[1:] -
                                             ref.ptr[:-1]).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
    weights = configs_weight * configs_forces_weight

    log_var = torch.log(pred["forces_var"])

    raw_mse = torch.square(ref["forces"] - pred["forces"])
    raw_nll = 0.5 * (raw_mse / pred["forces_var"] + log_var)

    return reduce_loss(weights * raw_mse,
                       ddp), reduce_loss(weights * raw_nll, ddp)


def mean_normed_error_forces(ref: Batch,
                             pred: TensorDict,
                             ddp: Optional[bool] = None) -> torch.Tensor:
    raw_loss = torch.linalg.vector_norm(ref["forces"] - pred["forces"],
                                        ord=2,
                                        dim=-1)
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Dipole Loss Function
# ------------------------------------------------------------------------------


def weighted_mean_squared_error_dipole(
        ref: Batch,
        pred: TensorDict,
        ddp: Optional[bool] = None) -> torch.Tensor:
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
    raw_loss = torch.square((ref["dipole"] - pred["dipole"]) / num_atoms)
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Polarizability Loss Function
# ------------------------------------------------------------------------------


def weighted_mean_squared_error_polarizability(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[
        bool] = None,  # ,mean: Optional[torch.Tensor] = None , std: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # polarizability: [n_graphs, ]
    # ref_polar = ref["polarizability"].view(-1, 3, 3) * std.view(1, 3, 3) + mean.view(1, 3, 3) if mean is not None and std is not None else ref["polarizability"]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)  # [n_graphs,1]
    raw_loss = torch.square(
        (ref["polarizability"].view(-1, 3, 3) - pred["polarizability"]) /
        num_atoms)
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Conditional Losses for Forces
# ------------------------------------------------------------------------------


def conditional_mse_forces(ref: Batch,
                           pred: TensorDict,
                           ddp: Optional[bool] = None) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(ref.weight, ref.ptr[1:] -
                                             ref.ptr[:-1]).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
    # Define multiplication factors for different regimes.
    factors = torch.tensor([1.0, 0.7, 0.4, 0.1],
                           device=ref["forces"].device,
                           dtype=ref["forces"].dtype)
    err = ref["forces"] - pred["forces"]
    se = torch.zeros_like(err)
    norm_forces = torch.norm(ref["forces"], dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    se[c1] = torch.square(err[c1]) * factors[0]
    se[c2] = torch.square(err[c2]) * factors[1]
    se[c3] = torch.square(err[c3]) * factors[2]
    se[~(c1 | c2 | c3)] = torch.square(err[~(c1 | c2 | c3)]) * factors[3]
    raw_loss = configs_weight * configs_forces_weight * se
    return reduce_loss(raw_loss, ddp)


def conditional_huber_forces(
    ref_forces: torch.Tensor,
    pred_forces: torch.Tensor,
    huber_delta: float,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    factors = huber_delta * torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=ref_forces.device, dtype=ref_forces.dtype)
    norm_forces = torch.norm(ref_forces, dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    c4 = ~(c1 | c2 | c3)
    se = torch.zeros_like(pred_forces)
    se[c1] = torch.nn.functional.huber_loss(ref_forces[c1],
                                            pred_forces[c1],
                                            reduction="none",
                                            delta=factors[0])
    se[c2] = torch.nn.functional.huber_loss(ref_forces[c2],
                                            pred_forces[c2],
                                            reduction="none",
                                            delta=factors[1])
    se[c3] = torch.nn.functional.huber_loss(ref_forces[c3],
                                            pred_forces[c3],
                                            reduction="none",
                                            delta=factors[2])
    se[c4] = torch.nn.functional.huber_loss(ref_forces[c4],
                                            pred_forces[c4],
                                            reduction="none",
                                            delta=factors[3])
    return reduce_loss(se, ddp)


# ------------------------------------------------------------------------------
# Loss Modules Combining Multiple Quantities
# ------------------------------------------------------------------------------


class WeightedEnergyForcesLoss(torch.nn.Module):

    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        return self.energy_weight * loss_energy + self.forces_weight * loss_forces

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})")


class WeightedEnergyForcesGaussianNLLLoss(torch.nn.Module):

    def __init__(self,
                 energy_weight=1.0,
                 forces_weight=1.0,
                 energy_uncertainty_weight=1.0,
                 forces_uncertainty_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "energy_uncertainty_weight",
            torch.tensor(energy_uncertainty_weight,
                         dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_uncertainty_weight",
            torch.tensor(forces_uncertainty_weight,
                         dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        loss_mse_energy, loss_nll_energy = weighted_gaussian_nll_energy(
            ref, pred, ddp=ddp)
        loss_mse_forces, loss_nll_forces = gaussian_nll_forces(ref,
                                                               pred,
                                                               ddp=ddp)
        return (self.energy_weight * loss_mse_energy +
                self.forces_weight * loss_mse_forces +
                self.energy_uncertainty_weight * loss_nll_energy +
                self.forces_uncertainty_weight * loss_nll_forces)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"energy_uncertainty_weight={self.energy_uncertainty_weight:.3f}, "
            f"forces_uncertainty_weight={self.forces_uncertainty_weight:.3f})")


class WeightedForcesLoss(torch.nn.Module):

    def __init__(self, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        return self.forces_weight * loss_forces

    def __repr__(self):
        return f"{self.__class__.__name__}(forces_weight={self.forces_weight:.3f})"


class WeightedForcesGaussianNLLLoss(torch.nn.Module):

    def __init__(self,
                 forces_weight: float = 1.0,
                 forces_uncertainty_weight: float = 1.0) -> None:
        super().__init__()
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_uncertainty_weight",
            torch.tensor(forces_uncertainty_weight,
                         dtype=torch.get_default_dtype()),
        )

    def forward(
        self,
        ref: Batch,
        pred: TensorDict,
        ddp: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss_mse, loss_nll = gaussian_nll_forces(ref, pred, ddp=ddp)
        return self.forces_weight * loss_mse + self.forces_nll_weight * loss_nll

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"forces_weight={self.forces_weight.item():.3f}, "
            f"forces_uncertainty_weight={self.forces_uncertainty_weight.item():.3f})"
        )


class WeightedEnergyForcesStressLoss(torch.nn.Module):

    def __init__(self,
                 energy_weight=1.0,
                 forces_weight=1.0,
                 stress_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_stress = weighted_mean_squared_stress(ref, pred, ddp)
        return (self.energy_weight * loss_energy +
                self.forces_weight * loss_forces +
                self.stress_weight * loss_stress)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedEnergyForcesStressGaussianNLLLoss(torch.nn.Module):

    def __init__(self,
                 energy_weight=1.0,
                 forces_weight=1.0,
                 stress_weight=1.0,
                 energy_uncertainty_weight=1.0,
                 forces_uncertainty_weight=1.0,
                 stress_uncertainty_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "energy_uncertainty_weight",
            torch.tensor(energy_uncertainty_weight,
                         dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_uncertainty_weight",
            torch.tensor(forces_uncertainty_weight,
                         dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_uncertainty_weight",
            torch.tensor(stress_uncertainty_weight,
                         dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        loss_mse_energy, loss_nll_energy = weighted_gaussian_nll_energy(
            ref, pred, ddp=ddp)
        loss_mse_forces, loss_nll_forces = gaussian_nll_forces(ref,
                                                               pred,
                                                               ddp=ddp)
        loss_mse_stress, loss_nll_stress = weighted_gaussian_nll_stress(
            ref, pred, ddp=ddp)
        return (self.energy_weight * loss_mse_energy +
                self.forces_weight * loss_mse_forces +
                self.stress_weight * loss_mse_stress +
                self.energy_uncertainty_weight * loss_nll_energy +
                self.forces_uncertainty_weight * loss_nll_forces +
                self.stress_uncertainty_weight * loss_nll_stress)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"stress_weight={self.stress_weight:.3f}, "
            f"energy_uncertainty_weight={self.energy_uncertainty_weight:.3f}, "
            f"forces_uncertainty_weight={self.forces_uncertainty_weight:.3f}, "
            f"stress_uncertainty_weight={self.stress_uncertainty_weight:.3f})")


class WeightedHuberEnergyForcesStressLoss(torch.nn.Module):

    def __init__(self,
                 energy_weight=1.0,
                 forces_weight=1.0,
                 stress_weight=1.0,
                 huber_delta=0.01) -> None:
        super().__init__()
        # We store the huber_delta rather than a loss with fixed reduction.
        self.huber_delta = huber_delta
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        if ddp:
            loss_energy = torch.nn.functional.huber_loss(
                ref["energy"] / num_atoms,
                pred["energy"] / num_atoms,
                reduction="none",
                delta=self.huber_delta,
            )
            loss_energy = reduce_loss(loss_energy, ddp)
            loss_forces = torch.nn.functional.huber_loss(
                ref["forces"],
                pred["forces"],
                reduction="none",
                delta=self.huber_delta)
            loss_forces = reduce_loss(loss_forces, ddp)
            loss_stress = torch.nn.functional.huber_loss(
                ref["stress"],
                pred["stress"],
                reduction="none",
                delta=self.huber_delta)
            loss_stress = reduce_loss(loss_stress, ddp)
        else:
            loss_energy = torch.nn.functional.huber_loss(
                ref["energy"] / num_atoms,
                pred["energy"] / num_atoms,
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_forces = torch.nn.functional.huber_loss(
                ref["forces"],
                pred["forces"],
                reduction="mean",
                delta=self.huber_delta)
            loss_stress = torch.nn.functional.huber_loss(
                ref["stress"],
                pred["stress"],
                reduction="mean",
                delta=self.huber_delta)
        return (self.energy_weight * loss_energy +
                self.forces_weight * loss_forces +
                self.stress_weight * loss_stress)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class UniversalLoss(torch.nn.Module):

    def __init__(self,
                 energy_weight=1.0,
                 forces_weight=1.0,
                 stress_weight=1.0,
                 huber_delta=0.01) -> None:
        super().__init__()
        self.huber_delta = huber_delta
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
        configs_energy_weight = ref.energy_weight
        configs_forces_weight = torch.repeat_interleave(
            ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
        if ddp:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="none",
                delta=self.huber_delta,
            )
            loss_energy = reduce_loss(loss_energy, ddp)
            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )
            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="none",
                delta=self.huber_delta,
            )
            loss_stress = reduce_loss(loss_stress, ddp)
        else:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )
            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="mean",
                delta=self.huber_delta,
            )
        return (self.energy_weight * loss_energy +
                self.forces_weight * loss_forces +
                self.stress_weight * loss_stress)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedEnergyForcesVirialsLoss(torch.nn.Module):

    def __init__(self,
                 energy_weight=1.0,
                 forces_weight=1.0,
                 virials_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "virials_weight",
            torch.tensor(virials_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_virials = weighted_mean_squared_virials(ref, pred, ddp)
        return (self.energy_weight * loss_energy +
                self.forces_weight * loss_forces +
                self.virials_weight * loss_virials)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, virials_weight={self.virials_weight:.3f})"
        )


class WeightedEnergyForcesVirialsGaussianNLLLoss(torch.nn.Module):

    def __init__(self,
                 energy_weight=1.0,
                 forces_weight=1.0,
                 virials_weight=1.0,
                 energy_uncertainty_weight=1.0,
                 forces_uncertainty_weight=1.0,
                 virials_uncertainty_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "virials_weight",
            torch.tensor(virials_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "energy_uncertainty_weight",
            torch.tensor(energy_uncertainty_weight,
                         dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_uncertainty_weight",
            torch.tensor(forces_uncertainty_weight,
                         dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "virials_uncertainty_weight",
            torch.tensor(virials_uncertainty_weight,
                         dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        loss_mse_energy, loss_nll_energy = weighted_gaussian_nll_energy(
            ref, pred, ddp=ddp)
        loss_mse_forces, loss_nll_forces = gaussian_nll_forces(ref,
                                                               pred,
                                                               ddp=ddp)
        loss_mse_virials, loss_nll_virials = weighted_gaussian_nll_virials(
            ref, pred, ddp=ddp)
        return (self.energy_weight * loss_mse_energy +
                self.forces_weight * loss_mse_forces +
                self.virials_weight * loss_mse_virials +
                self.energy_uncertainty_weight * loss_nll_energy +
                self.forces_uncertainty_weight * loss_nll_forces +
                self.virials_uncertainty_weight * loss_nll_virials)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"virials_weight={self.virials_weight:.3f}, "
            f"energy_uncertainty_weight={self.energy_uncertainty_weight:.3f}, "
            f"forces_uncertainty_weight={self.forces_uncertainty_weight:.3f}, "
            f"virials_uncertainty_weight={self.virials_uncertainty_weight:.3f})"
        )


class DipoleSingleLoss(torch.nn.Module):

    def __init__(self, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        loss = (weighted_mean_squared_error_dipole(ref, pred, ddp) * 100.0
                )  # scale adjustment
        return self.dipole_weight * loss

    def __repr__(self):
        return f"{self.__class__.__name__}(dipole_weight={self.dipole_weight:.3f})"


class DipolePolarLoss(torch.nn.Module):

    def __init__(
        self,
        dipole_weight=1.0,
        polarizability_weight=1.0
    ) -> (
            None
    ):  # dipole_mean=None,dipole_std=None,polarizability_mean=None,polarizability_std=None
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
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
        loss_dipole = weighted_mean_squared_error_dipole(
            ref, pred, ddp
        )  # ,self.dipole_mean,self.dipole_std) #* 100.0  # scale adjustment

        loss_polarizability = weighted_mean_squared_error_polarizability(
            ref, pred, ddp
        )  # ,self.polarizability_mean,self.polarizability_std) #* 100.0  # scale adjustment
        return (self.dipole_weight * loss_dipole +
                self.polarizability_weight * loss_polarizability)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dipole_weight={self.dipole_weight:.3f}, polarizability_weight={self.polarizability_weight:.3f})"
        )


class WeightedEnergyForcesDipoleLoss(torch.nn.Module):

    def __init__(self,
                 energy_weight=1.0,
                 forces_weight=1.0,
                 dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_dipole = weighted_mean_squared_error_dipole(ref, pred,
                                                         ddp) * 100.0
        return (self.energy_weight * loss_energy +
                self.forces_weight * loss_forces +
                self.dipole_weight * loss_dipole)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.dipole_weight:.3f})"
        )


class WeightedEnergyForcesL1L2Loss(torch.nn.Module):

    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self,
                ref: Batch,
                pred: TensorDict,
                ddp: Optional[bool] = None) -> torch.Tensor:
        loss_energy = weighted_mean_absolute_error_energy(ref, pred, ddp)
        loss_forces = mean_normed_error_forces(ref, pred, ddp)
        return self.energy_weight * loss_energy + self.forces_weight * loss_forces

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})")
