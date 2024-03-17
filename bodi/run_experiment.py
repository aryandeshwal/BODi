import inspect
import pickle
import time
import warnings
from copy import deepcopy
from typing import Dict, Optional, Union, List

import torch
import numpy as np
from torch import Tensor
from torch.quasirandom import SobolEngine
from botorch import fit_gpytorch_model
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.models import SingleTaskGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.fully_bayesian import MIN_INFERRED_NOISE_LEVEL
from botorch.models.model import ModelList
from botorch.models.transforms import Normalize, Standardize
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.utils.warnings import NumericalWarning

from .dictionary_kernel import DictionaryKernel
from .categorical_dictionary_kernel import DictionaryKernel as CatDictionaryKernel
from .pestcontrol import PestControl

from .optimize import optimize_acq_function_mixed_alternating, \
                             optimize_acqf_binary_local_search, \
                             optimize_acqf_categorical_local_search, \
                             convert_to_ohe
from .test_functions import LABS, SVM, Ackley53, MaxSAT60


def run_experiment(
    n_replications: int, base_seed: int = 1234, save_to_pickle: bool = True, fname: Optional[str] = None, **kwargs,
) -> Dict[str, Union[str, Optional[Tensor]]]:
    Xs, Ys, metadata = [], [], []
    for i in range(n_replications):
        print(f"=== Replication {i + 1}/{n_replications} ===")
        X, Y, meta = _run_single_trial(torch_seed=base_seed + i, **kwargs)
        print(f"best value = {torch.max(Y):.3f}")
        if save_to_pickle:
            fname = (
                fname
                or "./"
                + kwargs["evalfn"]
                + "_n0="
                + str(kwargs["n_initial_points"])
                + "_n="
                + str(kwargs["max_evals"])
                + "_q="
                + str(kwargs["batch_size"])
                + ".pkl"
            )
            pickle.dump((Xs, Ys, metadata), open(fname, "wb"))
            print(f"Results saved to: {fname}")

        Xs.append(X)
        Ys.append(Y)
        metadata.append(meta)
    Xs, Ys = torch.stack(Xs), torch.stack(Ys)

    if save_to_pickle:
        fname = (
            fname
            or "./"
            + kwargs["evalfn"]
            + "_n0="
            + str(kwargs["n_initial_points"])
            + "_n="
            + str(kwargs["max_evals"])
            + "_q="
            + str(kwargs["batch_size"])
            + ".pkl"
        )
        pickle.dump((Xs, Ys, metadata), open(fname, "wb"))
        print(f"Results saved to: {fname}")
    return Xs, Ys, metadata


def _run_single_trial(
    torch_seed: int,
    evalfn: str,
    max_evals: int,
    n_initial_points: int,
    batch_size: int = 1,
    n_binary: int = 0,
    n_categorical: int = 0,
    n_continuous: int = 0,
    category_sizes: List[int] = [],
    # category_size: int = 5,
    init_with_k_spaced_binary_sobol: bool = True,
    n_prototype_vectors: int = 10,
    verbose: bool = False,
    feature_costs: Optional[Tensor] = None,
) -> Dict[str, Union[str, Optional[Tensor]]]:
    _run_single_trial_input_kwargs = deepcopy(inspect.getargvalues(inspect.currentframe())[-1])
    start_time = time.time()
    torch.manual_seed(torch_seed)  # For reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tkwargs = {"dtype": torch.double, "device": device}
    if verbose:
        print(tkwargs)

    if evalfn == "LABS":
        assert n_categorical == 0, "LABS has no categorical parameters"
        assert n_continuous == 0, "LABS has no continuous parameters"
        assert n_binary > 0, "LABS need a non-zero number of binary parameters"
        f = LABS(n_binary=n_binary, **tkwargs)
    elif evalfn == "MaxSAT60":
        f = MaxSAT60(n_binary=n_binary, **tkwargs)
        assert n_binary == 60, "MaxSAT60 defined for 60 binary variables"
        assert n_categorical == 0, "MaxSAT60 has no categorical parameters"
        assert n_continuous == 0, "MaxSAT60 has no continuous parameters"
    elif evalfn == "PestControl":
        f = PestControl(**tkwargs)
        assert n_categorical == 25, "PestControl defined for 25 categorical variables"
        assert n_binary == 0, "PestControl defined for 0 binary variables"
        assert n_continuous == 0, "PestControl defined for 0 continuous variables"
    elif evalfn == "Ackley53":
        f = Ackley53(**tkwargs)
        assert n_binary == 50, "Ackley53 defined for 50 binary variables"
        assert n_continuous == 3, "Ackley53 defined for 3 continuous variables"
        assert n_categorical == 0, "Ackley53 has no categorical parameters"
    elif evalfn == "SVM":
        assert feature_costs is not None
        f = SVM(n_features=n_binary, feature_costs=feature_costs)
        reference_point = torch.tensor([1.0, 1.1 * feature_costs.sum()], **tkwargs)
        assert n_binary > 0, "SVM defined for >0 binary variables"
        assert n_continuous == 3, "SVM defined for 3 continuous variables"
        assert n_categorical == 0, "SVM has no categorical parameters"
    else:
        raise ValueError(f"Unknown evalfn {evalfn}")

    # Get initial Sobol points
    X = SobolEngine(dimension=f.dim, scramble=True, seed=torch_seed).draw(n_initial_points).to(**tkwargs)
    if init_with_k_spaced_binary_sobol:
        X[:, f.binary_inds] = 0
        with torch.random.fork_rng():
            torch.manual_seed(torch_seed)
            k = torch.randint(low=1, high=n_binary - 1, size=(n_initial_points,), device=device)
            binary_inds = torch.tensor(f.binary_inds, device=device)
            for i in range(n_initial_points):
                X[i, binary_inds[torch.randperm(n_binary)][: k[i]]] = 1

    # Rescale the Sobol points
    X = f.bounds[0] + (f.bounds[1] - f.bounds[0]) * X
    X[:, f.binary_inds] = X[:, f.binary_inds].round()  # Round binary variables
    # assert f.n_categorical == 0, "TODO"
    if evalfn == "PestControl":
        X_per_dim = []
        for i in range(len(category_sizes)):
            X_per_dim.append(torch.randint(0, category_sizes[i], (1, n_initial_points), **tkwargs))
        X = torch.cat(X_per_dim).T
        X_ohe = convert_to_ohe(X, category_sizes=category_sizes)
        print(X.shape)
    Y = torch.tensor([f(x) for x in X]).to(**tkwargs)
    assert Y.ndim == 2 if evalfn == "SVM" else Y.ndim == 1

    afo_config = {
        "n_initial_candts": 2000,
        "n_restarts": 20,
        "afo_init_design": "random",
        "n_alternate_steps": 50,
        "num_cmaes_steps": 50,
        "num_ls_steps": 50,
        "n_spray_points": 200,
        "verbose": False,
        "add_spray_points": True,
        "n_binary": n_binary,
        "n_cont": n_continuous,
        "category_sizes":category_sizes,
        # "category_size":category_size,
        "n_categorical":f.n_categorical,
    }
    while len(X) < max_evals:
        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(torch.tensor(0.9, **tkwargs), torch.tensor(10.0, **tkwargs)),
            noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
        )
        if evalfn == 'PestControl':
            dictionary_kernel = DictionaryKernel(
                num_basis_vectors=n_prototype_vectors,
                binary_dims=np.arange(X_ohe.shape[-1]),
                num_dims=X_ohe.shape[-1],
                similarity=True,
            )
            # dictionary_kernel = CatDictionaryKernel(
            #     num_basis_vectors=n_prototype_vectors,
            #     categorical_dims=f.categorical_dims,
            #     num_dims=f.dim,
            #     similarity=True,
            # )
        else:
            dictionary_kernel = DictionaryKernel(
                num_basis_vectors=n_prototype_vectors,
                binary_dims=f.binary_inds,
                num_dims=f.dim,
                similarity=True,
            )
        covar_module = ScaleKernel(
            base_kernel=dictionary_kernel,
            outputscale_prior=GammaPrior(torch.tensor(2.0, **tkwargs), torch.tensor(0.15, **tkwargs)),
            outputscale_constraint=GreaterThan(1e-6)
        )
        train_Y = (Y - Y.mean()) / Y.std() if evalfn != "SVM" else (Y[:, 0] - Y[:, 0].mean()) / Y[:, 0].std()
        if evalfn == 'PestControl':
            gp_model = SingleTaskGP(
                train_X=X_ohe,
                train_Y=train_Y.unsqueeze(-1),
                covar_module=covar_module,
                input_transform=Normalize(d=X_ohe.shape[-1]),
                likelihood=likelihood,
            )
        else:
            gp_model = SingleTaskGP(
                train_X=X,
                train_Y=train_Y.unsqueeze(-1),
                covar_module=covar_module,
                input_transform=Normalize(d=X.shape[-1]),
                likelihood=likelihood,
            )
        mll = ExactMarginalLogLikelihood(model=gp_model, likelihood=gp_model.likelihood)
        fit_gpytorch_model(mll)

        if evalfn == "SVM":

            def compute_feature_costs(x):
                return (x[..., f.binary_inds] * feature_costs).sum(dim=-1, keepdims=True)

            weights = torch.tensor([-1, -1], **tkwargs)
            deterministic_model = GenericDeterministicModel(f=compute_feature_costs)
            pareto_points = X[is_non_dominated(Y * weights)].clone()
            model_list = ModelList(gp_model, deterministic_model)

            objective = WeightedMCMultiOutputObjective(weights=weights)
            sampler = SobolQMCNormalSampler(num_samples=128, collapse_batch_dims=True)
            acqf = qNoisyExpectedHypervolumeImprovement(
                X_baseline=X,
                model=model_list,
                ref_point=objective(reference_point),
                objective=objective,
                sampler=sampler,
            )
        elif batch_size == 1:
            acqf = ExpectedImprovement(model=gp_model, best_f=train_Y.max())
            pareto_points = X[torch.argmax(train_Y)].unsqueeze(0).clone()
        else:
            acqf = qExpectedImprovement(model=gp_model, best_f=train_Y.max())
            pareto_points = X[torch.argmax(train_Y)].unsqueeze(0).clone()

        with warnings.catch_warnings():  # Filter jitter warnings
            warnings.filterwarnings("ignore", category=NumericalWarning)
            if n_binary > 0 and n_continuous == 0:
                next_x, acq_val = optimize_acqf_binary_local_search(
                    acqf, afo_config=afo_config, pareto_points=pareto_points, q=batch_size
                )
            elif n_binary > 0 and n_continuous > 0:  # mixed search space
                cont_dims = torch.arange(n_binary, n_binary + n_continuous, device=device)
                next_x, acq_val = optimize_acq_function_mixed_alternating(
                    acqf, cont_dims=cont_dims, pareto_points=pareto_points, q=batch_size, afo_config=afo_config,
                )
            elif n_binary == 0 and n_continuous == 0 and n_categorical > 0:
                next_x, acq_val = optimize_acqf_categorical_local_search(
                    acqf, afo_config=afo_config, pareto_points=pareto_points, q=batch_size,
                )


        X = torch.cat([X, next_x])
        X_ohe = convert_to_ohe(X, category_sizes=category_sizes)
        Y = torch.cat([Y, torch.tensor([f(x) for x in next_x], **tkwargs)])
        if verbose:
            if evalfn == "SVM":
                Y_pareto_old = Y[:-1].clone()
                Y_pareto_old = Y_pareto_old[is_non_dominated(Y_pareto_old * weights)]
                Y_pareto_old = Y_pareto_old[Y_pareto_old[:, 1].argsort(), :]
                Y_pareto_new = Y.clone()
                Y_pareto_new = Y_pareto_new[is_non_dominated(Y_pareto_new * weights)]
                Y_pareto_new = Y_pareto_new[Y_pareto_new[:, 1].argsort(), :]
                if len(Y_pareto_old) != len(Y_pareto_new) or not (Y_pareto_old == Y_pareto_new).all():
                    print(f"\nPareto frontier after {len(X) + 1} iterations:")
            else:
                print(
                    f"best value @ {len(X)} = {torch.max(Y):.3f}  | at {torch.argmax(Y)}, "
                    f"best point no of ones @ {torch.sum(X[torch.argmax(Y)])}, "
                    f"unique x's = {len(torch.unique(X, dim=0))}, "
                    f"acq value = {acq_val.item():.2e}"
                )

    # We only want max_evals evaluation in case we did too many
    X, Y = X[:max_evals], Y[:max_evals]
    # Save the results to manifold
    end_time = time.time()
    metadata = {
        "total_time": end_time - start_time,
        "kwargs": _run_single_trial_input_kwargs,
    }
    return X, Y, metadata
