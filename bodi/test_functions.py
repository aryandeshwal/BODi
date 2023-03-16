import math
import os
import pathlib
from abc import abstractmethod

import numpy as np
import torch
from sklearn.svm import SVR
from torch import Tensor
from xgboost import XGBRegressor
import pandas as pd
from copy import deepcopy

class TestFunction:
    """
    The abstract class for all benchmark functions acting as objective functions for BO.
    Note that we assume all problems will be minimization problem, so convert maximisation problems as appropriate.
    """

    # this should be changed if we are tackling a mixed, or continuous problem, for e.g.
    problem_type = "categorical"

    def __init__(self, normalize=True, **kwargs):
        self.normalize = normalize
        self.n_vertices = None
        self.config = None
        self.dim = None
        self.continuous_dims = None
        self.categorical_dims = None
        self.int_constrained_dims = None

    def _check_int_constrained_dims(self):
        if self.int_constrained_dims is None:
            return
        assert self.continuous_dims is not None, (
            "int_constrained_dims must be a subset of the continuous_dims, " "but continuous_dims is not supplied!"
        )
        int_dims_np = np.asarray(self.int_constrained_dims)
        cont_dims_np = np.asarray(self.continuous_dims)
        assert np.all(np.in1d(int_dims_np, cont_dims_np)), (
            "all continuous dimensions with integer "
            "constraint must be themselves contained in the "
            "continuous_dimensions!"
        )

    @abstractmethod
    def compute(self, x, normalize=None):
        raise NotImplementedError()

    def sample_normalize(self, size=None):
        if size is None:
            size = 2 * self.dim + 1
        y = []
        for _ in range(size):
            x = np.array([np.random.choice(self.config[_]) for _ in range(self.dim)])
            y.append(self.compute(x, normalize=False,))
        y = np.array(y)
        return np.mean(y), np.std(y)

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)


class _MaxSAT(TestFunction):
    def __init__(self, data_filename, random_seed=None, normalize=False, **kwargs):
        super(_MaxSAT, self).__init__(normalize, **kwargs)
        base_path = os.path.dirname(os.path.realpath(__file__))
        f = open(os.path.join(base_path, "data/", data_filename), "rt")
        line_str = f.readline()
        while line_str[:2] != "p ":
            line_str = f.readline()
        self.n_variables = int(line_str.split(" ")[2])
        self.n_clauses = int(line_str.split(" ")[3])
        self.n_vertices = np.array([2] * self.n_variables)
        self.config = self.n_vertices
        clauses = [(float(clause_str.split(" ")[0]), clause_str.split(" ")[1:-1]) for clause_str in f.readlines()]
        f.close()
        weights = np.array([elm[0] for elm in clauses]).astype(np.float32)
        weight_mean = np.mean(weights)
        weight_std = np.std(weights)
        self.weights = (weights - weight_mean) / weight_std
        self.clauses = [
            ([abs(int(elm)) - 1 for elm in clause], [int(elm) > 0 for elm in clause]) for _, clause in clauses
        ]

    def compute(self, x, normalize=None):
        if not isinstance(x, torch.Tensor):
            try:
                x = torch.tensor(x.astype(int))
            except:
                raise Exception("Unable to convert x to a pytorch tensor!")
        return self.evaluate(x)

    def evaluate(
        self, x,
    ):
        assert x.numel() == self.n_variables
        if x.dim() == 2:
            x = x.squeeze(0)
        x_np = (x.cpu() if x.is_cuda else x).numpy().astype(bool)
        satisfied = np.array([(x_np[clause[0]] == clause[1]).any() for clause in self.clauses])
        return np.sum(self.weights * satisfied) * x.float().new_ones(1, 1)


class MaxSAT60(_MaxSAT):
    def __init__(self, n_binary, **tkwargs):
        super().__init__(data_filename="frb-frb10-6-4.wcnf")
        self.n_binary = n_binary
        self.binary_inds = list(range(n_binary))
        self.n_continuous = 0
        self.continuous_inds = []
        self.n_categorical = 0
        self.categorical_inds = []
        self.dim = n_binary
        self.bounds = torch.stack((torch.zeros(n_binary), torch.ones(n_binary))).to(**tkwargs)

class Ackley53(TestFunction):
    problem_type = "mixed"

    # Taken and adapted from the the MVRSM codebase
    def __init__(self, lamda=1e-6, normalize=False, **tkwargs):
        super(Ackley53, self).__init__(normalize)
        self.n_binary = 50
        self.binary_inds = list(range(self.n_binary))
        self.n_continuous = 3
        self.continuous_inds = [50, 51, 52]
        self.n_categorical = 0
        self.categorical_inds = []
        self.dim = self.n_binary + self.n_continuous
        self.bounds = torch.stack((torch.zeros(self.dim), torch.ones(self.dim))).to(**tkwargs)
        self.n_vertices = 2 * np.ones(len(self.binary_inds), dtype=int)
        self.config = self.n_vertices
        self.lamda = lamda
        # specifies the range for the continuous variables
        # self.lb, self.ub = np.array([-1, -1, -1]), np.array([+1, +1, +1])
        self.feature_idxs = torch.arange(50)

    @staticmethod
    def _ackley(X):
        a = 20
        b = 0.2
        c = 2 * np.pi
        sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(np.square(X), axis=1) / 53))
        cos_term = -1 * np.exp(np.sum(np.cos(c * np.copy(X)) / 53, axis=1))
        result = a + np.exp(1) + sum_sq_term + cos_term
        return result

    def compute(self, X, normalize=None):
        if type(X) == torch.Tensor:
            X = X.numpy()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # To make sure there is no cheating, round the discrete variables before calling the function
        X[:, self.binary_inds] = np.round(X[:, self.binary_inds])
        X[:, self.continuous_inds] = -1 + 2 * X[:, self.continuous_inds]
        result = self._ackley(X)
        return -1*(result + self.lamda * np.random.rand(*result.shape))[0]


class LABS(object):
    def __init__(self, n_binary, **tkwargs):
        self.n_binary = n_binary
        self.binary_inds = list(range(n_binary))
        self.n_continuous = 0
        self.continuous_inds = []
        self.n_categorical = 0
        self.categorical_inds = []
        self.dim = n_binary
        self.bounds = torch.stack((torch.zeros(n_binary), torch.ones(n_binary))).to(**tkwargs)

    def __call__(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return torch.tensor([self._evaluate_single(xx) for xx in x]).to(x)

    def _evaluate_single(self, x_eval):
        x = deepcopy(x_eval)
        assert x.dim() == 1
        if x.dim() == 2:
            x = x.squeeze(0)
        assert x.shape[0] == self.n_binary
        x = x.cpu().numpy()
        N = x.shape[0]
        x[x == 0] = -1.
        # print(f'x transformed {x}')
        E = 0  # energy
        for k in range(1, N):
            C_k = 0
            for j in range(0, N - k):
                C_k += (x[j] * x[j + k])
            E += C_k ** 2
        if E == 0:
            print("found zero")
        return (N**2)/ (2 * E)

def load_uci_data(seed, n_features):
    try:
        path = str(pathlib.Path(__file__).parent.resolve()) + "/data/slice_localization_data.csv"
        df = pd.read_csv(path, sep=",")
    except:
        raise ValueError(
            "Failed to load `slice_localization_data.csv`. The slice dataset can be downloaded "
            "from: https://archive.ics.uci.edu/ml/datasets/Relative+location+of+CT+slices+on+axial+axis"
        )
    data = df.to_numpy()

    # Get the input data
    X = data[:, :-1]
    X -= X.min(axis=0)
    X = X[:, X.max(axis=0) > 1e-6]  # Throw away constant dimensions
    X = X / (X.max(axis=0) - X.min(axis=0))
    X = 2 * X - 1
    assert X.min() == -1 and X.max() == 1

    # Standardize targets
    y = data[:, -1]
    y = (y - y.mean()) / y.std()

    # Only keep 10,000 data points and n_features features
    shuffled_indices = np.random.RandomState(0).permutation(X.shape[0])[:10_000]  # Use seed 0
    X, y = X[shuffled_indices], y[shuffled_indices]

    # Use Xgboost to figure out feature importances and keep only the most important features
    xgb = XGBRegressor(max_depth=8).fit(X, y)
    inds = (-xgb.feature_importances_).argsort()
    X = X[:, inds[:n_features]]

    # Train/Test split on a subset of the data
    train_n = int(math.floor(0.50 * X.shape[0]))
    train_x, train_y = X[:train_n], y[:train_n]
    test_x, test_y = X[train_n:], y[train_n:]

    return train_x, train_y, test_x, test_y


class SVM:
    def __init__(self, n_features: int, feature_costs: Tensor, **tkwargs):
        self.train_x, self.train_y, self.test_x, self.test_y = load_uci_data(seed=0, n_features=n_features)
        self.n_binary = n_features
        self.binary_inds = list(range(n_features))
        self.n_continuous = 3
        self.continuous_inds = list(range(n_features, n_features + 3))
        self.n_categorical = 0
        self.categorical_inds = []
        self.dim = n_features + 3
        self.bounds = torch.stack((torch.zeros(self.dim), torch.ones(self.dim))).to(**tkwargs)
        assert feature_costs.shape == (n_features,) and feature_costs.min() >= 0
        self.feature_costs = feature_costs

    def __call__(self, x: Tensor):
        assert x.shape == (self.dim,)
        assert (x >= self.bounds[0]).all() and (x <= self.bounds[1]).all()
        assert ((x[self.binary_inds] == 0) | (x[self.binary_inds] == 1)).all()  # Features must be 0 or 1
        inds_selected = np.where(x[self.binary_inds].cpu().numpy() == 1)[0]
        if len(inds_selected) == 0:  # Silly corner case with no features
            rmse, feature_cost = 1.0, 0.0
        else:
            epsilon = 0.01 * 10 ** (2 * x[-3])  # Default = 0.1
            C = 0.01 * 10 ** (4 * x[-2])  # Default = 1.0
            gamma = (1 / self.n_binary) * 0.1 * 10 ** (2 * x[-1])  # Default = 1.0 / self.n_features
            model = SVR(C=C.item(), epsilon=epsilon.item(), gamma=gamma.item())
            model.fit(self.train_x[:, inds_selected], self.train_y)  #
            pred = model.predict(self.test_x[:, inds_selected])
            rmse = math.sqrt(((pred - self.test_y) ** 2).mean(axis=0).item())
            feature_cost = self.feature_costs[inds_selected].sum().item()
        return [rmse, feature_cost]
