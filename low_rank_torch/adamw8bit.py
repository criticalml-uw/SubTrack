import torch
from bitsandbytes.optim.optimizer import Optimizer2State

from .low_rank_projector import LowRankProjector


class AdamW8bit(Optimizer2State):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, optim_bits=32,
                 args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        super().__init__("adam", params, lr, betas, eps, weight_decay, 8, args, min_8bit_size, percentile_clipping,
                         block_wise, is_paged=is_paged)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        overflows = []

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  # needed for fairseq pure fp16 training
            self.initialized = True

        # if self.is_paged: self.page_mng.prefetch_all()
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                matrix = None
                # GaLore Projection
                if "rank" in group:
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

                    if 'weight_decay' in group and group['weight_decay'] > 0:
                        # ensure that the weight decay is not applied to the norm grad
                        group['weight_decay_saved'] = group['weight_decay']
                        group['weight_decay'] = 0

                    grad, matrix = state["projector"].project(p.grad, state["step"])
                    
                    if group.get("adaptive_optimizer", False):
                        update_step = state["step"] % state["projector"].subspace_update_interval == 0
                        projection_updated = update_step and state["step"] > 0

                        if projection_updated:
                            if state["projector"].prev_ortho_matrix is None:
                                state["projector"].prev_ortho_matrix = state["projector"]._copy()

                            beta2 = group["betas"][1]
                            rank_on_rows = grad.shape[0] == state["projector"].rank

                            if "exp_avg" not in state:
                                state["exp_avg"] = grad.clone().float().zero_()
                                state["exp_avg_sq"] = grad.clone().float().zero_()

                            exp_avg = state["exp_avg"]
                            exp_avg_sq = state["exp_avg_sq"]

                            if rank_on_rows:
                                C = state["projector"].ortho_matrix.T @ state["projector"].prev_ortho_matrix
                                C_exp_avg = C @ exp_avg
                                state["exp_avg_sq"] = (1.0 - beta2) * (
                                    (C.square() @ (exp_avg_sq - exp_avg.square())) + (C_exp_avg).square()
                                ).abs()
                                state["exp_avg"] = C_exp_avg
                            else:
                                C = state["projector"].prev_ortho_matrix @ state["projector"].ortho_matrix.T
                                exp_avg_C = exp_avg @ C
                                state["exp_avg_sq"] = (1.0 - beta2) * (
                                    ((exp_avg_sq - exp_avg.square()) @ C.square()) + (exp_avg_C).square()
                                ).abs()
                                state["exp_avg"] = exp_avg_C

                            del C
                    
                    beta1, beta2 = group["betas"]
                    if "exp_avg" not in state:
                        state["exp_avg"] = grad.clone().float().zero_()
                        state["exp_avg_sq"] = grad.clone().float().zero_()

                    state["exp_avg"].mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                    state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    denom = state["exp_avg_sq"].sqrt().add_(group["eps"])
                    norm_grad = state["exp_avg"] / denom

                    # suboptimal implementation
                    p.saved_data = p.data.clone()
                    p.data = grad.clone().to(p.data.dtype).to(p.data.device)
                    p.data.zero_()
                    p.grad = grad

                if 'state1' not in state:
                    self.init_state(group, p, gindex, pindex)

                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)
                torch.cuda.synchronize()
                
                # GaLore Projection Back
                if "rank" in group:
                    # p.data = p.saved_data.add_(state["projector"].project_back(p.data))
                    if matrix is not None:
                        if group.get("recovery_scaling", False):
                            subgrad = state["projector"].project_back(p.grad)
                            norm_dim = 0 if norm_grad.shape[0] < norm_grad.shape[1] else 1
                            scaling_factor = (
                                torch.norm(norm_grad, dim=norm_dim) /
                                (torch.norm(p.grad, dim=norm_dim) + 1e-8)
                            )
                            if norm_dim == 1:
                                scaling_factor = scaling_factor.unsqueeze(1)
                            scaling_grad = (p.grad - subgrad) * scaling_factor

                            if "scaling_grad" in state:
                                scaling_grad_norm = torch.norm(scaling_grad)
                                limiter = max(
                                    scaling_grad_norm / (state["scaling_grad"] + 1e-8), 1.01
                                ) / 1.01
                                scaling_grad = scaling_grad / limiter
                                state["scaling_grad"] = scaling_grad_norm / limiter
                            else:
                                state["scaling_grad"] = torch.norm(scaling_grad)

                            norm_grad = state["projector"].project_back(norm_grad.to(state["projector"].ortho_matrix.dtype)) + + scaling_grad.to(state["projector"].ortho_matrix.dtype)
                        else:
                            norm_grad = state["projector"].project_back(norm_grad.to(state["projector"].ortho_matrix.dtype))

                        # apply weight decay
                        if 'weight_decay_saved' in group:
                            p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay_saved'])
                            group['weight_decay'] = group['weight_decay_saved']
                            del group['weight_decay_saved']

        if self.is_paged:
            # all paged operation are asynchronous, we need
            # to sync to make sure all tensors are in the right state
            torch.cuda.synchronize()

        return loss
