from typing import Any, List, Optional

import torch
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from torch import Tensor

def encode_via_basis_vectors(x: Tensor, basis_vectors: Tensor, similarity: bool = True):
    # x has shape n x (q) x d
    # basis_vectors has shape m x d
    # returns tensor of size n x (q) x m
    # return (x @ basis_vectors.T)/x.shape[0]
    dists = (x.unsqueeze(-3) != (basis_vectors.unsqueeze(-2))).transpose(-2, -3)
    return 1 - dists.to(x).mean(-1) if similarity else dists.to(x).mean(-1)

### uncomment below lines 16-57 for binary wavelet dictionary and comment lines 61 to 67
# def get_bft_matrix(n, **tkwargs):
#     if n==2:
#         return torch.tensor([[1., 1.], [1., 0.]])
#     if n == 4:
#         return torch.tensor([[1., 1., 1., 1.], [1., 1., 0., 0.], [1., 0., 1., 1.], [1., 0., 1., 0.]])
#     elif n == 8:
#         return torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1.],
#                 [1., 1., 1., 1., 1., 1., 0., 0.],
#                 [1., 1., 0., 0., 0., 0., 1., 1.],
#                 [1., 1., 0., 0., 1., 1., 0., 0.],
#                 [1., 1., 0., 1., 0., 0., 1., 1.],
#                 [1., 1., 0., 1., 0., 1., 0., 0.],
#                 [1., 0., 1., 0., 1., 0., 1., 1.],
#                 [1., 0., 1., 0., 1., 0., 1., 0.],])
#     B_ul = torch.cat([torch.ones(2, 2, **tkwargs), torch.ones(2, n-4, **tkwargs)], axis=1)
#     B_ul = torch.cat([B_ul, torch.cat([torch.ones(n-4, 2, **tkwargs), 1-get_bft_matrix(n-4)], axis=1)], axis=0)
#     assert B_ul.shape == (n-2, n-2)
#     B_ur = torch.ones(n-2, 2, **tkwargs)
#     B_ur[1::2] = 0.
#     B_ll = B_ur.T
#     B_lr = torch.tensor([[1., 1.], [1., 0.]], **tkwargs)
#     BFT = torch.cat([torch.cat([B_ul, B_ur], axis=1), torch.cat([B_ll, B_lr], axis=1)], axis=0)
#     return BFT


# def generate_random_basis_vector(num_basis_vectors, num_binary, **tkwargs):
#     # print(f'wavelet design {num_basis_vectors}')
#     # BFT = get_bft_matrix(col_size, **tkwargs)
#     if num_binary % 2 == 1:
#         BFT = get_bft_matrix(num_binary-1, **tkwargs)
#         cp_idxs = np.sort(np.random.choice(np.arange(num_binary-1), (num_basis_vectors,), replace=False))
#     else:
#         BFT = get_bft_matrix(num_binary, **tkwargs)
#         cp_idxs = np.sort(np.random.choice(np.arange(num_binary), (num_basis_vectors,), replace=False))
#     # print(cp_idxs)
#     BFT = BFT[:, cp_idxs]
#     # BFT = (BFT[:, :num_basis_vectors])
#     if BFT.shape[0] == num_binary:
#         return BFT.T
#     else:
#         BFT = torch.cat([torch.ones(1, num_basis_vectors, **tkwargs), BFT], axis=0).T
#         return BFT


### Diverse random dictionary generation (recommended default choice)
def generate_random_basis_vector(num_basis_vectors: int, num_binary: int, **tkwargs: Any):
    k = torch.randint(low=1, high=num_binary - 1, size=(num_basis_vectors,))
    fs_dict_vectors = torch.zeros(num_basis_vectors, num_binary, **tkwargs)
    for i in range(len(fs_dict_vectors)):
        fs_dict_vectors[i, torch.randperm(num_binary)[: k[i]]] = 1
    return fs_dict_vectors


class DictionaryKernel(Kernel):
    has_lengthscale = False

    def __init__(
        self,
        binary_dims: List[int],
        num_dims: int,
        num_basis_vectors: int = 5,
        basis_vectors: Optional[Tensor] = None,
        similarity=True,
        skip: bool = False,  # if true, adds binary vars to continuous group, in addition to the embedding
        additive: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert len(set(binary_dims)) == len(binary_dims) and min(binary_dims) >= 0 and max(binary_dims) <= num_dims - 1
        if basis_vectors is None:
            basis_vectors = generate_random_basis_vector(
                num_basis_vectors=num_basis_vectors, num_binary=len(binary_dims)
            )
        num_basis_vectors = len(basis_vectors)
        num_bin_dims = len(binary_dims)
        num_cont_dims = num_dims - num_bin_dims
        ard_num_dims = len(basis_vectors) + num_cont_dims
        if skip:  # since raw binary dims are also added to continuous group
            ard_num_dims += num_bin_dims
        self.base_kernel = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims, **kwargs)
        self.basis_vectors = basis_vectors
        self.cont_dims = list(set(range(num_dims)).difference(set(binary_dims)))
        self.binary_dims = binary_dims
        self.similarity = similarity
        self.skip = skip
        self.additive = additive
        if additive:
            self.base_kernel = ScaleKernel(self.base_kernel)
            self.additive_binary_kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=1)) # len(binary_dims)))

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False,) -> Tensor:
        b1 = x1[..., self.binary_dims]
        encoded_vecs_1 = encode_via_basis_vectors(
            x=b1, basis_vectors=self.basis_vectors.to(x1), similarity=self.similarity,
        )
        c1 = x1 if self.skip else x1[..., self.cont_dims]
        z1 = torch.cat([encoded_vecs_1, c1], axis=-1)

        b2 = x2[..., self.binary_dims]
        encoded_vecs_2 = encode_via_basis_vectors(
            x=b2, basis_vectors=self.basis_vectors.to(x1), similarity=self.similarity,
        )
        c2 = x2 if self.skip else x2[..., self.cont_dims]
        z2 = torch.cat([encoded_vecs_2, c2], axis=-1)
        res = self.base_kernel.forward(z1, z2, diag=diag, last_dim_is_batch=last_dim_is_batch)
        if self.additive:
            res = res + self.additive_binary_kernel(b1, b2, diag=diag, last_dim_is_batch=last_dim_is_batch)
        return res
