# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from transformers.utils.versions import require_version

from .low_rank_projector import LowRankProjector
from low_rank_torch.subspace_evaluation_analyzer import SubspaceEvaluationAnalyzer


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            no_deprecation_warning: bool = False,
            adaptive_optimizer: bool = False
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias, "adaptive_optimizer": adaptive_optimizer,}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0
                
                matrix = None
                if "rank" in group:           
                    if group.get("rand_proj", False):
                        # GaLore Projection
                        # if "rank" in group:
                        if "projector" not in state:
                            state["projector"] = LowRankProjector(
                                group["rank"], scale=group["scale"],
                                proj_type=group["proj_type"],
                                st_init_step_size=group["st_init_step_size"],
                                subspace_update_method=group["subspace_update_method"],
                                st_step_size_scheduler=group["st_step_size_scheduler"],
                                st_step_size_coef=group["st_step_size_coef"],
                                st_noise_sigma2=group["st_noise_sigma2"],
                                st_subspace_coef=group["st_subspace_coef"],
                                subspace_update_interval=group["subspace_update_interval"]
                            )
                        
                        rand = (state["step"] >= group["rand_epoch"])
                        grad, matrix = state["projector"].project(grad, state["step"], rand)

                    else:
                        # GaLore Projection
                        if "projector" not in state:
                            state["projector"] = LowRankProjector(
                                group["rank"], scale=group["scale"],
                                proj_type=group["proj_type"],
                                st_init_step_size=group["st_init_step_size"],
                                subspace_update_method=group["subspace_update_method"],
                                st_step_size_scheduler=group["st_step_size_scheduler"],
                                st_step_size_coef=group["st_step_size_coef"],
                                st_noise_sigma2=group["st_noise_sigma2"],
                                st_subspace_coef=group["st_subspace_coef"],
                                subspace_update_interval=group["subspace_update_interval"]
                            )

                        grad, matrix = state["projector"].project(grad, state["step"], rand=False)
                        
                        
                        # module_name = group["module_names"][i]
                        # if state["step"] % state["projector"].update_proj_gap == 0:
                        #     SubspaceEvaluationAnalyzer.save_gradient_subspace(
                        #         state["projector"].ortho_matrix, state["step"],
                        #         module_name, group["rank"], state["projector"].update_proj_gap
                        #     )                       
                    
                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                
                # GoLore
                if matrix is not None:
                    if "matrix" in state:
                        if state["projector"].proj_type == "left":
                            exp_avg = matrix.T @ (state["matrix"] @ exp_avg)
                        else:
                            exp_avg = (exp_avg @ state["matrix"]) @ matrix.T
                    state["matrix"] = matrix 
                
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom
            
                # GaLore Projection Back
                # Norm Scaling
                if "rank" in group:
                    update_step = state["step"] % state["projector"].subspace_update_interval == 0
                    projection_updated = update_step and state["step"] > 0 
                    
                    if projection_updated and group.get("adaptive_optimizer", False):
                        
                        if state["projector"].prev_ortho_matrix is None:
                            state["projector"].prev_ortho_matrix = state["projector"]._copy()
                        
                        # beta2 = group["betas"][1]
                                
                        rank_on_rows = state["exp_avg"].shape[0] == state["projector"].rank # True = left-sided
                        if rank_on_rows:
                            # Left-sided update
                            # Change-of-basis matrix C 
                            C = state["projector"].ortho_matrix.t() @ state["projector"].prev_ortho_matrix  
                                    
                            C_exp_avg = C @ state["exp_avg"] 
                            state["exp_avg_sq"] = (1.0 - beta2) * (
                                (C.square() @ (state["exp_avg_sq"] - state["exp_avg"].square())) + (C_exp_avg).square()
                                ).abs()
                        
                            state["exp_avg"] = C_exp_avg
                            
                            del C, C_exp_avg
                        else:
                            # Right-sided update
                            # Change-of-basis matrix C 
                            C = state["projector"].prev_ortho_matrix @ state["projector"].ortho_matrix.t() 
                            
                            exp_avg_C = state["exp_avg"] @ C 
                            state["exp_avg_sq"] = (1.0 - beta2) * (
                                ((state["exp_avg_sq"] - state["exp_avg"].square()) @ C.square()) + (exp_avg_C).square()
                                ).abs()  
                            
                            state["exp_avg"] = exp_avg_C
                            
                            del C, exp_avg_C  
                               
                    if group.get("recovery_scaling", False):
                            subgrad = state["projector"].project_back(grad)
                            norm_dim = 0 if norm_grad.shape[0] < norm_grad.shape[1] else 1
                            scaling_factor = (
                                torch.norm(norm_grad, dim=norm_dim) /
                                (torch.norm(grad, dim=norm_dim) + 1e-8)
                            )
                            if norm_dim == 1:
                                scaling_factor = scaling_factor.unsqueeze(1)
                            scaling_grad = (p.grad - subgrad) * scaling_factor
                            
                            # Norm-Growth Limiter
                            if not group.get("norm_growth_limiter_off", False):
                                if "scaling_grad" in state:
                                    scaling_grad_norm = torch.norm(scaling_grad)
                                    lim = group.get("norm_growth_limit", 1.01)
                                    limiter = max(
                                        scaling_grad_norm / (state["scaling_grad"] + 1e-8),
                                        lim,
                                    ) / lim
                                    scaling_grad = scaling_grad / limiter
                                    state["scaling_grad"] = scaling_grad_norm / limiter
                             
                                else:
                                    state["scaling_grad"] = torch.norm(scaling_grad)
                            
                            norm_grad = state["projector"].project_back(norm_grad) + scaling_grad
                    else:
                        norm_grad = state["projector"].project_back(norm_grad)

                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss
