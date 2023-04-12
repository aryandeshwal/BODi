from typing import Any, List

import torch
from gpytorch.kernels import Kernel, MaternKernel
from torch import Tensor
import numpy as np



def encode_via_random_vectors(x: Tensor, random_basis_vectors: Tensor, similarity: bool = True):
    # x has shape n x (q) x d
    # random_basis_vectors has shape m x d
    # returns tensor of size n x (q) x m
    # return (x @ random_basis_vectors.T)/x.shape[0]
    dists = (x.unsqueeze(-3) != (random_basis_vectors.unsqueeze(-2))).transpose(-2, -3)
    return 1 - dists.to(x).mean(-1) if similarity else dists.to(x).mean(-1)

def runif_in_simplex(n):
  '''generates uniformly random vector in the n-simplex '''
  k = np.random.exponential(scale=1.0, size=n)
  return k / sum(k)

def generate_random_basis_vector(num_basis_vectors: int, input_dim: int, ncatgs=5, **tkwargs: Any):
    fs_dict_vectors = []
    for _ in range(num_basis_vectors):
        simplex_sample = runif_in_simplex(ncatgs)
        k = (simplex_sample[:-1] * input_dim).astype(np.int32)
        k = np.append(k, input_dim - np.sum(k))
        basis_vector = torch.zeros(input_dim, **tkwargs)
        perm_idxs = torch.randperm(input_dim)
        for j in range(ncatgs):
            if k[j] == 0:
                continue
            else:
                basis_vector[perm_idxs[np.sum(k[:j]): np.sum(k[:j]) + k[j]]] = j
        fs_dict_vectors.append(basis_vector)
    return torch.cat([x.unsqueeze(0) for x in fs_dict_vectors])


class DictionaryKernel(Kernel):
    has_lengthscale = False

    def __init__(
        self,
        categorical_dims: List[int],
        num_dims: int,
        num_basis_vectors: int = 5,
        similarity=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # assert len(set(binary_dims)) == len(binary_dims) and min(binary_dims) >= 0 and max(binary_dims) <= num_dims - 1
        basis_vectors = generate_random_basis_vector(num_basis_vectors=num_basis_vectors, input_dim=len(categorical_dims))
        num_cont_dims = num_dims - len(categorical_dims)
        self.base_kernel = MaternKernel(nu=2.5, ard_num_dims=len(basis_vectors) + num_cont_dims, **kwargs)
        self.basis_vectors = basis_vectors
        self.cont_dims = list(set(range(num_dims)).difference(set(categorical_dims)))
        self.categorical_dims = categorical_dims
        self.similarity = similarity

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        encoded_vecs = encode_via_random_vectors(
            x=x1[..., self.categorical_dims],
            random_basis_vectors=self.basis_vectors.to(x1),
            similarity=self.similarity,
        )
        x1 = torch.cat([encoded_vecs, x1[..., self.cont_dims]], axis=-1)
        encoded_vecs = encode_via_random_vectors(
            x=x2[..., self.categorical_dims],
            random_basis_vectors=self.basis_vectors.to(x1),
            similarity=self.similarity,
        )
        x2 = torch.cat([encoded_vecs, x2[..., self.cont_dims]], axis=-1)
        res = self.base_kernel.forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch)
        return res
