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
    mse = torch.square(ref["energy"] - pred["energy"])
    return reduce_loss(mse, ddp)


def gaussian_nll_energy(ref: Batch,
                        pred: TensorDict,
                        ddp: Optional[bool] = None) -> torch.Tensor:
    mse = torch.square(ref["energy"] - pred["energy"])
    nll = 0.5 * (mse / pred["energy_var"] + torch.log(pred["energy_var"]))
    return reduce_loss(nll, ddp)


def weighted_mean_squared_error_energy(
        ref: Batch,
        pred: TensorDict,
        ddp: Optional[bool] = None) -> torch.Tensor:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    mse = torch.square((ref["energy"] - pred["energy"]) / num_atoms)
    return reduce_loss(ref.weight * ref.energy_weight * mse, ddp)


def weighted_gaussian_nll_energy(ref: Batch,
                                 pred: TensorDict,
                                 ddp: Optional[bool] = None) -> torch.Tensor:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    var_per_atom = pred["energy_var"] / (num_atoms**2)
    mse = torch.square((ref["energy"] - pred["energy"]) / num_atoms)
    nll = 0.5 * (mse / var_per_atom + torch.log(var_per_atom))
    return reduce_loss(ref.weight * ref.energy_weight * nll, ddp)


def weighted_mean_absolute_error_energy(
        ref: Batch,
        pred: TensorDict,
        ddp: Optional[bool] = None) -> torch.Tensor:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    mae = torch.abs((ref["energy"] - pred["energy"]) / num_atoms)
    return reduce_loss(ref.weight * ref.energy_weight * mae, ddp)


def weighted_laplace_nll_energy(ref: Batch,
                                pred: TensorDict,
                                ddp: Optional[bool] = None) -> torch.Tensor:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    b_per_atom = torch.sqrt(pred["forces_var"] / 2.0) / num_atoms
    mae = torch.abs((ref["energy"] - pred["energy"]) / num_atoms)
    nll = mae / b_per_atom + torch.log(b_per_atom)
    return reduce_loss(ref.weight * ref.energy_weight * nll, ddp)


# ------------------------------------------------------------------------------
# Stress and Virials Loss Functions
# ------------------------------------------------------------------------------


def weighted_mean_squared_stress(ref: Batch,
                                 pred: TensorDict,
                                 ddp: Optional[bool] = None) -> torch.Tensor:
    weights = ref.weight.view(-1, 1, 1) * ref.stress_weight.view(-1, 1, 1)
    mse = torch.square(ref["stress"] - pred["stress"])
    return reduce_loss(weights * mse, ddp)


def weighted_gaussian_nll_stress(ref: Batch,
                                 pred: TensorDict,
                                 ddp: Optional[bool] = None) -> torch.Tensor:
    weights = ref.weight.view(-1, 1, 1) * ref.stress_weight.view(-1, 1, 1)
    mse = torch.square(ref["stress"] - pred["stress"])
    nll = 0.5 * (mse / pred["stress_var"] + torch.log(pred["stress_var"]))
    return reduce_loss(weights * nll, ddp)


def weighted_mean_squared_virials(ref: Batch,
                                  pred: TensorDict,
                                  ddp: Optional[bool] = None) -> torch.Tensor:
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)
    weights = ref.weight.view(-1, 1, 1) * ref.virials_weight.view(-1, 1, 1)
    mse = torch.square((ref["virials"] - pred["virials"]) / num_atoms)
    return reduce_loss(weights * mse, ddp)


def weighted_gaussian_nll_virials(ref: Batch,
                                  pred: TensorDict,
                                  ddp: Optional[bool] = None) -> torch.Tensor:
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)
    weights = ref.weight.view(-1, 1, 1) * ref.virials_weight.view(-1, 1, 1)
    var_per_atom = pred["virials_var"] / (num_atoms**2)
    mse = torch.square((ref["virials"] - pred["virials"]) / num_atoms)
    nll = 0.5 * (mse / var_per_atom + torch.log(var_per_atom))
    return reduce_loss(weights * nll, ddp)


# ------------------------------------------------------------------------------
# Forces Loss Functions
# ------------------------------------------------------------------------------


def mean_squared_error_forces(ref: Batch,
                              pred: TensorDict,
                              ddp: Optional[bool] = None) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(ref.weight, ref.ptr[1:] -
                                             ref.ptr[:-1]).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
    mse = (torch.square(ref["forces"] - pred["forces"]))
    return reduce_loss(configs_weight * configs_forces_weight * mse, ddp)


def gaussian_nll_forces(ref: Batch,
                        pred: TensorDict,
                        ddp: Optional[bool] = None) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(ref.weight, ref.ptr[1:] -
                                             ref.ptr[:-1]).unsqueeze(-1)
    configs_f_weight = torch.repeat_interleave(ref.forces_weight, ref.ptr[1:] -
                                               ref.ptr[:-1]).unsqueeze(-1)
    mse = torch.square(ref["forces"] - pred["forces"])
    nll = 0.5 * (mse / pred["forces_var"] + torch.log(pred["forces_var"]))
    return reduce_loss(configs_weight * configs_f_weight * nll, ddp)


def mean_normed_error_forces(ref: Batch,
                             pred: TensorDict,
                             ddp: Optional[bool] = None) -> torch.Tensor:
    mae = torch.linalg.vector_norm(ref["forces"] - pred["forces"],
                                   ord=2,
                                   dim=-1)
    return reduce_loss(mae, ddp)


def laplace_nll_normed_forces(ref: Batch,
                              pred: TensorDict,
                              ddp: Optional[bool] = None) -> torch.Tensor:
    b = torch.sqrt(pred["forces_var"] / 2.0)
    mae = torch.linalg.vector_norm(ref["forces"] - pred["forces"],
                                   ord=2,
                                   dim=-1)
    nll = (mae / b) + b
    return reduce_loss(nll, ddp)


# ------------------------------------------------------------------------------
# Dipole Loss Function
# ------------------------------------------------------------------------------


def weighted_mean_squared_error_dipole(
        ref: Batch,
        pred: TensorDict,
        ddp: Optional[bool] = None) -> torch.Tensor:
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
    mse = torch.square((ref["dipole"] - pred["dipole"]) / num_atoms)
    return reduce_loss(mse, ddp)


def weighted_gaussian_nll_dipole(ref: Batch,
                                 pred: TensorDict,
                                 ddp: Optional[bool] = None) -> torch.Tensor:
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
    var_per_atom = pred["dipole_var"] / (num_atoms**2)
    mse = torch.square((ref["dipole"] - pred["dipole"]) / num_atoms)
    nll = 0.5 * (mse / var_per_atom + torch.log(var_per_atom))
    return reduce_loss(nll, ddp)


# ------------------------------------------------------------------------------
# Polarizability Loss Function
# ------------------------------------------------------------------------------


def weighted_mean_squared_error_polarizability(
        ref: Batch,
        pred: TensorDict,
        ddp: Optional[bool] = None) -> torch.Tensor:
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)
    mse = torch.square(
        (ref["polarizability"].view(-1, 3, 3) - pred["polarizability"]) /
        num_atoms)
    return reduce_loss(mse, ddp)


def weighted_gaussian_nll_polarizability(
        ref: Batch,
        pred: TensorDict,
        ddp: Optional[bool] = None) -> torch.Tensor:
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)
    var_per_atom = pred["polarizability_var"] / (num_atoms**2)
    mse = torch.square(
        (ref["polarizability"].view(-1, 3, 3) - pred["polarizability"]) /
        num_atoms)
    nll = 0.5 * (mse / var_per_atom + torch.log(var_per_atom))
    return reduce_loss(nll, ddp)


# ------------------------------------------------------------------------------
# Conditional Losses for Forces
# ------------------------------------------------------------------------------


def conditional_mse_forces(ref: Batch,
                           pred: TensorDict,
                           ddp: Optional[bool] = None) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(ref.weight, ref.ptr[1:] -
                                             ref.ptr[:-1]).unsqueeze(-1)
    configs_f_weight = torch.repeat_interleave(ref.forces_weight, ref.ptr[1:] -
                                               ref.ptr[:-1]).unsqueeze(-1)
    factors = torch.tensor([1.0, 0.7, 0.4, 0.1],
                           device=ref["forces"].device,
                           dtype=ref["forces"].dtype)
    err = ref["forces"] - pred["forces"]
    mse = torch.zeros_like(err)
    norm_forces = torch.norm(ref["forces"], dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    mse[c1] = torch.square(err[c1]) * factors[0]
    mse[c2] = torch.square(err[c2]) * factors[1]
    mse[c3] = torch.square(err[c3]) * factors[2]
    mse[~(c1 | c2 | c3)] = torch.square(err[~(c1 | c2 | c3)]) * factors[3]
    return reduce_loss(configs_weight * configs_f_weight * mse, ddp)


def conditional_gaussian_nll_forces(
        ref: Batch,
        pred: TensorDict,
        ddp: Optional[bool] = None) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(ref.weight, ref.ptr[1:] -
                                             ref.ptr[:-1]).unsqueeze(-1)
    configs_f_weight = torch.repeat_interleave(ref.forces_weight, ref.ptr[1:] -
                                               ref.ptr[:-1]).unsqueeze(-1)
    factors = torch.tensor([1.0, 0.7, 0.4, 0.1],
                           device=ref["forces"].device,
                           dtype=ref["forces"].dtype)
    err = ref["forces"] - pred["forces"]
    mse = torch.zeros_like(err)
    norm_forces = torch.norm(ref["forces"], dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    mse[c1] = torch.square(err[c1]) * factors[0]
    mse[c2] = torch.square(err[c2]) * factors[1]
    mse[c3] = torch.square(err[c3]) * factors[2]
    mse[~(c1 | c2 | c3)] = torch.square(err[~(c1 | c2 | c3)])
    nll = 0.5 * (mse / pred["forces_var"] + torch.log(pred["forces_var"]))
    return reduce_loss(configs_weight * configs_f_weight * factors[3] * nll,
                       ddp)


def conditional_huber_forces(
    ref: Batch,
    pred: TensorDict,
    huber_delta: float,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    configs_f_weight = torch.repeat_interleave(ref.forces_weight, ref.ptr[1:] -
                                               ref.ptr[:-1]).unsqueeze(-1)
    ref_f = configs_f_weight * ref["forces"]
    pred_f = configs_f_weight * pred["forces"]
    factors = huber_delta * torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=ref_f.device, dtype=ref_f.dtype)
    norm_f = torch.norm(ref_f, dim=-1)
    c1 = norm_f < 100
    c2 = (norm_f >= 100) & (norm_f < 200)
    c3 = (norm_f >= 200) & (norm_f < 300)
    c4 = ~(c1 | c2 | c3)
    he = torch.zeros_like(pred_f)
    he[c1] = torch.nn.functional.huber_loss(ref_f[c1],
                                            pred_f[c1],
                                            reduction="none",
                                            delta=factors[0])
    he[c2] = torch.nn.functional.huber_loss(ref_f[c2],
                                            pred_f[c2],
                                            reduction="none",
                                            delta=factors[1])
    he[c3] = torch.nn.functional.huber_loss(ref_f[c3],
                                            pred_f[c3],
                                            reduction="none",
                                            delta=factors[2])
    he[c4] = torch.nn.functional.huber_loss(ref_f[c4],
                                            pred_f[c4],
                                            reduction="none",
                                            delta=factors[3])
    return reduce_loss(he, ddp)


def conditional_huber_nll_forces(
    ref: Batch,
    pred: TensorDict,
    huber_delta: float,
    log_z_delta: torch.Tensor,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    configs_f_weight = torch.repeat_interleave(ref.forces_weight, ref.ptr[1:] -
                                               ref.ptr[:-1]).unsqueeze(-1)
    ref_f = configs_f_weight * ref["forces"]
    pred_f = configs_f_weight * pred["forces"]
    std = torch.sqrt(pred["forces_var"])
    inv_std = 1.0 / std
    factors = huber_delta * torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=ref_f.device, dtype=ref_f.dtype)
    norm_f = torch.norm(ref_f, dim=-1)
    c1 = norm_f < 100
    c2 = (norm_f >= 100) & (norm_f < 200)
    c3 = (norm_f >= 200) & (norm_f < 300)
    c4 = ~(c1 | c2 | c3)
    he = torch.zeros_like(pred_f)
    he[c1] = torch.nn.functional.huber_loss(ref_f[c1] * inv_std[c1],
                                            pred_f[c1] * inv_std[c1],
                                            reduction="none",
                                            delta=factors[0],
                                            weight=inv_std[c1])
    he[c2] = torch.nn.functional.huber_loss(ref_f[c2] * inv_std[c2],
                                            pred_f[c2] * inv_std[c2],
                                            reduction="none",
                                            delta=factors[1],
                                            weight=inv_std[c2])
    he[c3] = torch.nn.functional.huber_loss(ref_f[c3] * inv_std[c3],
                                            pred_f[c3] * inv_std[c3],
                                            reduction="none",
                                            delta=factors[2],
                                            weight=inv_std[c3])
    he[c4] = torch.nn.functional.huber_loss(ref_f[c4] * inv_std[c4],
                                            pred_f[c4] * inv_std[c4],
                                            reduction="none",
                                            delta=factors[3],
                                            weight=inv_std[c4])
    nll = he + torch.log(std) + log_z_delta
    return reduce_loss(nll, ddp)


def z_delta(delta: float) -> torch.Tensor:
    normal = torch.distributions.Normal(0, 1)
    cdf = torch.erf(delta / torch.sqrt(2))
    pdf = torch.exp(normal.log_prob(delta))
    return torch.sqrt(2 * torch.pi) * cdf + (delta) * pdf


def huber_energy(
    ref: Batch,
    pred: TensorDict,
    huber_delta: float,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    he = torch.nn.functional.huber_loss(
        ref["energy"] / num_atoms,
        pred["energy"] / num_atoms,
        reduction="none",
        delta=huber_delta,
    )
    if ddp:
        he = reduce_loss(he, ddp)
    else:
        he = he.mean()
    return he


def huber_nll_energy(
    ref: Batch,
    pred: TensorDict,
    huber_delta: float,
    log_z_delta: torch.Tensor,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    std_per_atom = torch.sqrt(pred["energy_var"]) / num_atoms
    inv_std_per_atom = 1.0 / std_per_atom
    nll = torch.nn.functional.huber_loss(
        ref["energy"] / num_atoms * inv_std_per_atom,
        pred["energy"] / num_atoms * inv_std_per_atom,
        reduction="none",
        delta=huber_delta,
        weight=inv_std_per_atom,
    ) + torch.log(std_per_atom) + log_z_delta
    if ddp:
        nll = reduce_loss(nll, ddp)
    else:
        nll = nll.mean()
    return nll


def weighted_huber_energy(
    ref: Batch,
    pred: TensorDict,
    huber_delta: float,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    he = torch.nn.functional.huber_loss(
        ref.energy_weight * ref["energy"] / num_atoms,
        ref.energy_weight * pred["energy"] / num_atoms,
        reduction="none",
        delta=huber_delta,
    )
    if ddp:
        he = reduce_loss(he, ddp)
    else:
        he = he.mean()
    return he


def weighted_huber_nll_energy(
    ref: Batch,
    pred: TensorDict,
    huber_delta: float,
    log_z_delta: torch.Tensor,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    std_per_atom = torch.sqrt(pred["energy_var"]) / num_atoms
    inv_std_per_atom = 1.0 / std_per_atom
    nll = torch.nn.functional.huber_loss(
        ref.energy_weight * ref["energy"] / num_atoms * inv_std_per_atom,
        ref.energy_weight * pred["energy"] / num_atoms * inv_std_per_atom,
        reduction="none",
        delta=huber_delta,
        weight=inv_std_per_atom,
    ) + torch.log(std_per_atom) + log_z_delta
    if ddp:
        nll = reduce_loss(nll, ddp)
    else:
        nll = nll.mean()
    return nll


def huber_forces(
    ref: Batch,
    pred: TensorDict,
    huber_delta: float,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    he = torch.nn.functional.huber_loss(
        ref["forces"],
        pred["forces"],
        reduction="none",
        delta=huber_delta,
    )
    if ddp:
        he = reduce_loss(he, ddp)
    else:
        he = he.mean()
    return he


def huber_nll_forces(
    ref: Batch,
    pred: TensorDict,
    huber_delta: float,
    log_z_delta: torch.Tensor,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    std = torch.sqrt(pred["forces_var"])
    inv_std = 1.0 / std
    nll = torch.nn.functional.huber_loss(
        ref["forces"] * inv_std,
        pred["forces"] * inv_std,
        reduction="none",
        delta=huber_delta,
        weight=inv_std,
    ) + torch.log(std) + log_z_delta
    if ddp:
        nll = reduce_loss(nll, ddp)
    else:
        nll = nll.mean()
    return nll


def huber_stress(
    ref: Batch,
    pred: TensorDict,
    huber_delta: float,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    he = torch.nn.functional.huber_loss(
        ref["stress"],
        pred["stress"],
        reduction="none",
        delta=huber_delta,
    )
    if ddp:
        he = reduce_loss(he, ddp)
    else:
        he = he.mean()
    return he


def huber_nll_stress(
    ref: Batch,
    pred: TensorDict,
    huber_delta: float,
    log_z_delta: torch.Tensor,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    std = torch.sqrt(pred["stress_var"])
    inv_std = 1.0 / std
    nll = torch.nn.functional.huber_loss(
        ref["stress"] * inv_std,
        pred["stress"] * inv_std,
        reduction="none",
        delta=huber_delta,
        weight=inv_std,
    ) + torch.log(std) + log_z_delta
    if ddp:
        nll = reduce_loss(nll, ddp)
    else:
        nll = nll.mean()
    return nll


def weighted_huber_stress(
    ref: Batch,
    pred: TensorDict,
    huber_delta: float,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    configs_s_weight = ref.stress_weight.view(-1, 1, 1)
    he = torch.nn.functional.huber_loss(
        configs_s_weight * ref["stress"],
        configs_s_weight * pred["stress"],
        reduction="none",
        delta=huber_delta,
    )
    if ddp:
        he = reduce_loss(he, ddp)
    else:
        he = he.mean()
    return he


def weighted_huber_nll_stress(
    ref: Batch,
    pred: TensorDict,
    huber_delta: float,
    log_z_delta: torch.Tensor,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
    std = torch.sqrt(pred["stress_var"])
    inv_std = 1.0 / std
    nll = torch.nn.functional.huber_loss(
        configs_stress_weight * ref["stress"] * inv_std,
        configs_stress_weight * pred["stress"] * inv_std,
        reduction="none",
        delta=huber_delta,
        weight=inv_std,
    ) + torch.log(std) + log_z_delta
    if ddp:
        nll = reduce_loss(nll, ddp)
    else:
        nll = nll.mean()
    return nll


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
        if pred["energy_var"] is not None and pred["force_var"] is not None:
            loss_energy = weighted_gaussian_nll_energy(ref, pred, ddp=ddp)
            loss_forces = gaussian_nll_forces(ref, pred, ddp=ddp)
        else:
            loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
            loss_forces = mean_squared_error_forces(ref, pred, ddp)
        return self.energy_weight * loss_energy + self.forces_weight * loss_forces

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})")


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
        if pred["force_var"] is not None:
            loss_forces = gaussian_nll_forces(ref, pred, ddp=ddp)
        else:
            loss_forces = mean_squared_error_forces(ref, pred, ddp)
        return self.forces_weight * loss_forces

    def __repr__(self):
        return f"{self.__class__.__name__}(forces_weight={self.forces_weight:.3f})"


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
        if pred["energy_var"] is not None and pred[
                "force_var"] is not None and pred["stress_var"] is not None:
            loss_energy = weighted_gaussian_nll_energy(ref, pred, ddp=ddp)
            loss_forces = gaussian_nll_forces(ref, pred, ddp=ddp)
            loss_stress = weighted_gaussian_nll_stress(ref, pred, ddp=ddp)
        else:
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
        if pred["energy_var"] is not None and pred[
                "force_var"] is not None and pred["stress_var"] is not None:
            log_z_delta = torch.log(z_delta(self.huber_delta))
            loss_energy = huber_nll_energy(ref, pred, self.huber_delta,
                                           log_z_delta, ddp)
            loss_forces = huber_nll_forces(ref, pred, self.huber_delta,
                                           log_z_delta, ddp)
            loss_stress = huber_nll_stress(ref, pred, self.huber_delta,
                                           log_z_delta, ddp)
        else:
            loss_energy = huber_nll_energy(ref, pred, self.huber_delta, ddp)
            loss_forces = huber_nll_forces(ref, pred, self.huber_delta, ddp)
            loss_stress = huber_nll_stress(ref, pred, self.huber_delta, ddp)
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
        if pred["energy_var"] is not None and pred[
                "force_var"] is not None and pred["stress_var"] is not None:
            log_z_delta = torch.log(z_delta(self.huber_delta))
            loss_energy = weighted_huber_nll_energy(ref, pred,
                                                    self.huber_delta,
                                                    log_z_delta, ddp)
            loss_forces = conditional_huber_nll_forces(ref, pred,
                                                       self.huber_delta,
                                                       log_z_delta, ddp)
            loss_stress = weighted_huber_nll_stress(ref, pred,
                                                    self.huber_delta,
                                                    log_z_delta, ddp)
        else:
            loss_energy = weighted_huber_energy(ref, pred, self.huber_delta,
                                                ddp)
            loss_forces = conditional_huber_forces(ref, pred, self.huber_delta,
                                                   ddp)
            loss_stress = weighted_huber_stress(ref, pred, self.huber_delta,
                                                ddp)
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
        if pred["energy_var"] is not None and pred[
                "force_var"] is not None and pred["virials_var"] is not None:
            loss_energy = weighted_gaussian_nll_energy(ref, pred, ddp=ddp)
            loss_forces = gaussian_nll_forces(ref, pred, ddp=ddp)
            loss_virials = weighted_gaussian_nll_virials(ref, pred, ddp=ddp)
        else:
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
        if pred["dipole_var"] is not None:
            loss = weighted_gaussian_nll_dipole(ref, pred, ddp) * 100.0
        else:
            loss = weighted_mean_squared_error_dipole(ref, pred, ddp) * 100.0
        return self.dipole_weight * loss

    def __repr__(self):
        return f"{self.__class__.__name__}(dipole_weight={self.dipole_weight:.3f})"


class DipolePolarLoss(torch.nn.Module):

    def __init__(self, dipole_weight=1.0, polarizability_weight=1.0) -> (None):
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
        if pred["dipole_var"] is not None and pred[
                "polarizability_var"] is not None:
            loss_dipole = weighted_gaussian_nll_dipole(ref, pred, ddp)
            loss_polarizability = weighted_gaussian_nll_polarizability(
                ref, pred, ddp)
        else:
            loss_dipole = weighted_mean_squared_error_dipole(ref, pred, ddp)
            loss_polarizability = weighted_mean_squared_error_polarizability(
                ref, pred, ddp)
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
        if pred["energy_var"] is not None and pred[
                "force_var"] is not None and pred["dipole_var"] is not None:
            loss_energy = weighted_gaussian_nll_energy(ref, pred, ddp)
            loss_forces = gaussian_nll_forces(ref, pred, ddp)
            loss_dipole = weighted_gaussian_nll_dipole(ref, pred, ddp) * 100.0
        else:
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
        if pred["energy_var"] is not None and pred["force_var"] is not None:
            loss_energy = weighted_laplace_nll_energy(ref, pred, ddp)
            loss_forces = laplace_nll_normed_forces(ref, pred, ddp)
        else:
            loss_energy = weighted_mean_absolute_error_energy(ref, pred, ddp)
            loss_forces = mean_normed_error_forces(ref, pred, ddp)
        return self.energy_weight * loss_energy + self.forces_weight * loss_forces

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})")
