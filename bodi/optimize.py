import contextlib
import os
from typing import Callable, Dict, Optional

import cma
import numpy as np
import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.acquisition import AcquisitionFunction
from gpytorch.kernels import Kernel, MaternKernel


def get_hamming_neighbors(x_discrete: torch.Tensor):
    r"""
    Returns all 1-hamming distance neighbors of a binary input tensor `x_discrete`.
    """
    X_loc = (x_discrete - torch.eye(x_discrete.shape[-1], dtype=x_discrete.dtype)).abs()
    return X_loc


def get_spray_points(pareto_points: torch.Tensor, n_binary: int, n_cont: int, n_spray_points: int):
    r"""
    Given a set of good points lying in the pareto frontier `pareto_points`, the method
    returns their perturbations (named spray points) computed by adding gaussian perturbation
    to the continuous parameters and 1-hamming distance neighbors of the binary parameters.

    Args:
        pareto_points: Tensor of best acquired points across BO run.
        n_binary: Number of binary parameters/input dimensions,
        n_cont: Number of continuous parameters/input dimensions,
    """
    perturb_nbors = None
    # TODO: make it faster by vectorization
    for x in pareto_points:
        nbds = get_hamming_neighbors(x[:n_binary])
        n_hamming_nbors = nbds.shape[0]
        cont_perturbs = x[n_binary:] + 1 / 8 * torch.randn(
            n_spray_points, n_cont
        )  # N(0, 1/64) to keep perturbations within ranges (assuming inputs scaled to [0, 1])
        assert cont_perturbs.shape == (n_spray_points, n_cont)
        cont_perturbs = cont_perturbs.clamp(0.0, 1.0)
        nbds = torch.cat(
            [
                nbds.repeat((n_spray_points, 1, 1)),
                cont_perturbs.repeat((nbds.shape[0], 1, 1)).transpose(0, 1),
            ],
            axis=-1,
        )
        assert nbds.shape == (n_spray_points, n_hamming_nbors, n_binary + n_cont)
        nbds = torch.reshape(nbds, (n_spray_points * n_hamming_nbors, n_binary + n_cont))
        if perturb_nbors is None:
            perturb_nbors = nbds
        else:
            perturb_nbors = torch.cat([perturb_nbors, nbds], axis=0)
    return perturb_nbors


def get_input_correct_order(X: torch.Tensor, cont_dims: torch.Tensor, discrete_dims: torch.Tensor):
    r"""
    Given an input in the format `batch_size x [[n_binary], [n_cont]]` where binary parameters
    are followed by continuous parameters, this method places the parameters in the correct indices
    given by `cont_dims` and `discrete_dims`.
    """
    n_binary = discrete_dims.shape[-1]
    swap_copy = X.clone()
    X[..., discrete_dims] = swap_copy[..., :n_binary]
    X[..., cont_dims] = swap_copy[..., n_binary:]
    return X


def optimize_acq_function_mixed_alternating(
    acq_function: AcquisitionFunction,
    cont_dims: torch.Tensor,
    pareto_points: torch.Tensor,
    afo_config: Dict,
    q: int = 1,
    n_initial_candts: int = 2000,
    n_restarts: int = 10,
    max_batch_size: int = 2048,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
):
    r"""
    Optimizes acquisition function over mixed binary and continuous input spaces.
    Multiple random restarting starting points are picked by evaluating a large set of initial candidates.
    From each starting point, alternating discrete local search and continuous
    optimization via (CMA-ES) is performed for a fixed number of iterations.
    All continuous parameters are assumed to be within [0, 1] bounds.

    Args:
        acq_function: BoTorch Acquisition function
        cont_dims: a tensor of indices corresponding to continuous parameters/inputs.
        pareto_points: Tensor of best acquired points across BO run
        afo_config: Configuration dictionary specifying following elements
                    "afo_init_design": random | equally spaced (linspace style),
                    "n_alternate_steps": number of alternating discrete and continuous search,
                    "num_cmaes_steps": number of steps for CMA-ES based continuous optimization,
                    "num_ls_steps": number of discrete local search steps,
                    "add_spray_points": whether to add candidates perturbed near best points,
                    "n_spray_points": number of candidates to perturb near best points,
        q: Number of candidates
        n_initial_candts: Number of initial candidates to select starting points,
        n_restarts: Number of random restarts,
        max_batch_size: The maximum number of choices to evaluate in batch.
            A large limit can cause excessive memory usage if the model has
            a large training set.
    """
    candidate_list = []
    base_X_pending = acq_function.X_pending if q > 1 else None

    tkwargs = {"device": pareto_points.device, "dtype": pareto_points.dtype}
    discrete_dims = torch.from_numpy(np.setdiff1d(np.arange(pareto_points.shape[-1]), cont_dims)).to(
        device=pareto_points.device
    )
    n_binary = discrete_dims.shape[-1]
    n_cont = cont_dims.shape[-1]
    afo_init_design = afo_config.get("afo_init_design", "random")

    for _ in range(q):
        if afo_init_design == "equally_spaced":
            # picking initial points by equally spaced number of features/binary inputs
            k = torch.linspace(1, n_binary, n_initial_candts, dtype=torch.int32)
            x_init_candts = SobolEngine(dimension=n_binary + n_cont, scramble=True).draw(n_initial_candts).to(**tkwargs)
            x_init_candts[:, :n_binary] = 0
            for i in range(len(x_init_candts)):
                x_init_candts[i][torch.randperm(n_binary)[: k[i]]] = 1
        else:
            x_init_candts = torch.cat(
                [
                    torch.randint(0, 2, (n_initial_candts, n_binary)),
                    torch.rand((n_initial_candts, n_cont)),
                ],
                axis=1,
            ).to(**tkwargs)

        add_spray_points = afo_config.get("add_spray_points", False)
        if add_spray_points is True:
            # append a set of neighbors perturbed from good points evaluated till now
            n_spray_points = afo_config.get("n_spray_points", 20)
            perturb_nbors = get_spray_points(pareto_points, n_binary, n_cont, n_spray_points)
            x_init_candts = torch.cat([x_init_candts, perturb_nbors], axis=0)

        with torch.no_grad():
            acq_init_candts = torch.cat(
                [
                    acq_function(get_input_correct_order(X_, cont_dims, discrete_dims).unsqueeze(1))
                    for X_ in x_init_candts.split(max_batch_size)
                ]
            )
        topk_indices = torch.topk(acq_init_candts, n_restarts)[1]
        best_X = x_init_candts[topk_indices]
        best_acq_val = acq_init_candts[topk_indices]
        n_alternate_steps = afo_config.get("n_alternate_steps", 10)
        for i in range(n_restarts):
            alternate_steps = 0
            while alternate_steps < n_alternate_steps:
                starting_acq_val = best_acq_val[i].clone()
                alternate_steps += 1
                # discrete search first
                num_ls_steps = afo_config.get("num_ls_steps", 10)
                for _ in range(num_ls_steps):
                    nbds = get_hamming_neighbors(best_X[i][:n_binary])
                    nbds = torch.cat(
                        [
                            nbds,
                            best_X[i][n_binary:].unsqueeze(0).repeat(nbds.shape[0], 1),
                        ],
                        axis=1,
                    )
                    with torch.no_grad():
                        acq_vals = acq_function(get_input_correct_order(nbds, cont_dims, discrete_dims).unsqueeze(1))
                    if torch.max(acq_vals) > best_acq_val[i]:
                        best_acq_val[i] = torch.max(acq_vals)
                        best_X[i] = nbds[torch.argmax(acq_vals)]
                    else:
                        break
                # CMA based continuous search
                cont_bounds = [[0] * (n_cont), [1] * (n_cont)]
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    es = cma.CMAEvolutionStrategy(
                        x0=best_X[i][n_binary:].numpy(),
                        sigma0=0.1,
                        inopts={"bounds": cont_bounds, "popsize": 50},
                    )
                    num_cmaes_steps = afo_config.get("num_cmaes_steps", 20)
                    n_cmaes_iter = 1
                    while not es.stop():
                        n_cmaes_iter += 1
                        xs = es.ask()
                        X = torch.cat(
                            [
                                best_X[i][:n_binary].unsqueeze(0).repeat(torch.tensor(xs).shape[0], 1),
                                torch.tensor(xs),
                            ],
                            axis=1,
                        ).to(**tkwargs)
                        # evaluate the acquisition function (optimizer assumes we're minimizing)
                        with torch.no_grad():
                            Y = (
                                -1
                                * acq_function(get_input_correct_order(X, cont_dims, discrete_dims).unsqueeze(1))
                                .detach()
                                .numpy()
                            )
                        es.tell(xs, Y)  # return the result to the optimizer
                        if n_cmaes_iter > num_cmaes_steps:
                            break
                if -1 * es.best.f > best_acq_val[i]:
                    best_X[i] = torch.cat([best_X[i][:n_binary], torch.from_numpy(es.best.x).float()])
                    best_acq_val[i] = -1 * es.best.f
                if (best_acq_val[i] - starting_acq_val) < 1e-6:
                    break  # out of the alternating continuous and discrete local search loop
        candidate_list.append(best_X[torch.argmax(best_acq_val)].unsqueeze(0))

        # set pending points
        candidates = torch.cat(candidate_list, dim=-2)
        if q > 1:
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2) if base_X_pending is not None else candidates
            )
    if q > 1:
        acq_function.set_X_pending(base_X_pending)

    if post_processing_func is not None:
        candidates = post_processing_func(candidates)

    with torch.no_grad():
        acq_value = acq_function(
            get_input_correct_order(candidates, cont_dims, discrete_dims)
        )  # compute joint acquisition value
    return candidates, acq_value


def get_neighbors(x_discrete, **tkwargs):
    X_loc = (x_discrete - torch.eye(x_discrete.shape[-1], **tkwargs)).abs()
    return X_loc


def optimize_acqf_binary_local_search(acqf, afo_config, pareto_points: torch.Tensor, q: int = 1):
    candidate_list = []
    base_X_pending = acqf.X_pending if q > 1 else None

    tkwargs = {"device": pareto_points.device, "dtype": pareto_points.dtype}
    n_initial_candts = afo_config["n_initial_candts"]  # 2000
    n_restarts = afo_config["n_restarts"]  # 5
    n_binary = afo_config["n_binary"]

    for _ in range(q):
        # picking initial points randomly
        x_init_candts = torch.randint(0, 2, (n_initial_candts, n_binary), **tkwargs)
        if afo_config["add_spray_points"] is True:
            # append a set of neighbors perturbed from good points evaluated till now
            perturb_nbors = None
            for x in pareto_points:
                nbds = get_neighbors(x[:n_binary], **tkwargs)
                if perturb_nbors is None:
                    perturb_nbors = nbds
                else:
                    perturb_nbors = torch.cat([perturb_nbors, nbds], axis=0)
            x_init_candts = torch.cat([x_init_candts, perturb_nbors], axis=0)

        with torch.no_grad():
            acq_init_candts = torch.cat([acqf(X_.unsqueeze(1)) for X_ in x_init_candts.split(16)])
        # print(f"x_init_candts {x_init_candts.shape}")
        topk_indices = torch.topk(acq_init_candts, n_restarts)[1]
        # print(f"topk_indices {topk_indices}")
        best_X = x_init_candts[topk_indices]
        best_acq_val = acq_init_candts[topk_indices]
        for i in range(n_restarts):
            num_ls_steps = afo_config["num_ls_steps"]  # number of local search steps
            for _ in range(num_ls_steps):
                nbds = get_neighbors(best_X[i][:n_binary], **tkwargs)
                with torch.no_grad():
                    acq_vals = acqf(nbds.unsqueeze(1))
                if torch.max(acq_vals) > best_acq_val[i]:
                    best_acq_val[i] = torch.max(acq_vals)
                    best_X[i] = nbds[torch.argmax(acq_vals)]
                else:
                    break
        candidate_list.append(best_X[torch.argmax(best_acq_val)].unsqueeze(0))

        # set pending points
        candidates = torch.cat(candidate_list, dim=-2)
        if q > 1:
            acqf.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2) if base_X_pending is not None else candidates
            )

    if q > 1:
        acqf.set_X_pending(base_X_pending)
    with torch.no_grad():
        acq_value = acqf(candidates)  # compute joint acquisition value
    return candidates, acq_value



def get_catg_neighbors(x_discrete, n_categories,  **tkwargs):
    X_loc = []
    for pt_idx in range(x_discrete.shape[0]):
        for i in range(x_discrete.shape[1]):
            for j in range(n_categories):
                if x_discrete[pt_idx][i] == j:
                    continue
                temp_x = x_discrete[pt_idx].clone()
                temp_x[i] = j
                X_loc.append(temp_x)
    return torch.cat([x.unsqueeze(0) for x in X_loc], dim=0).to(**tkwargs)

def optimize_acqf_categorical_local_search(acqf, afo_config, pareto_points: torch.Tensor, q: int = 1):
    candidate_list = []
    base_X_pending = acqf.X_pending if q > 1 else None

    tkwargs = {"device": pareto_points.device, "dtype": pareto_points.dtype}
    n_initial_candts = afo_config["n_initial_candts"]  # 2000
    n_restarts = afo_config["n_restarts"]  # 5
    input_dim = afo_config["n_categorical"]
    n_categories = afo_config["category_size"] # afo_config["n_categories"]

    for _ in range(q):
        # picking initial points randomly
        x_init_candts = torch.randint(0, n_categories, (n_initial_candts, input_dim), **tkwargs)
        if afo_config["add_spray_points"] is True:
            # append a set of neighbors perturbed from good points evaluated till now
            perturb_nbors = None
            for x in pareto_points:
                nbds = get_catg_neighbors(x.unsqueeze(0), n_categories, **tkwargs)
                # print(x, nbds)
                # print(f'nbds {nbds.shape}')
                if perturb_nbors is None:
                    perturb_nbors = nbds
                else:
                    # print(f'nbds {perturb_nbors.shape}') 
                    perturb_nbors = torch.cat([perturb_nbors, nbds], axis=0)
            x_init_candts = torch.cat([x_init_candts, perturb_nbors], axis=0)

        with torch.no_grad():
            acq_init_candts = torch.cat([acqf(X_.unsqueeze(1)) for X_ in x_init_candts.split(16)])
        # print(f"x_init_candts {x_init_candts.shape}")
        topk_indices = torch.topk(acq_init_candts, n_restarts)[1]
        # print(f"topk_indices {topk_indices}")
        best_X = x_init_candts[topk_indices]
        best_acq_val = acq_init_candts[topk_indices]
        for i in range(n_restarts):
            num_ls_steps = afo_config["num_ls_steps"]  # number of local search steps
            for _ in range(num_ls_steps):
                # print(f'best_X[i] {best_X[i].shape} {best_X[i].dtype}')
                nbds = get_catg_neighbors(best_X[i].unsqueeze(0), n_categories, **tkwargs)
                with torch.no_grad():
                    acq_vals = acqf(nbds.unsqueeze(1))
                if torch.max(acq_vals) > best_acq_val[i]:
                    best_acq_val[i] = torch.max(acq_vals)
                    best_X[i] = nbds[torch.argmax(acq_vals)]
                else:
                    break
        candidate_list.append(best_X[torch.argmax(best_acq_val)].unsqueeze(0))

        # set pending points
        candidates = torch.cat(candidate_list, dim=-2)
        if q > 1:
            acqf.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2) if base_X_pending is not None else candidates
            )

    if q > 1:
        acqf.set_X_pending(base_X_pending)
    with torch.no_grad():
        acq_value = acqf(candidates.unsqueeze(1)) # compute joint acquisition value
    return candidates, acq_value