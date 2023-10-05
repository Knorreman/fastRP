import math
from typing import List

import torch
from torch import Tensor
import random


class FastRP:

    edges_path: str
    embeddings_path: str
    k: int
    s: int
    d: int
    A: torch.sparse.Tensor
    L: torch.sparse.Tensor
    R: torch.sparse.Tensor
    adj_mat: torch.sparse

    def __init__(self,
                 edges_path: str,
                 embeddings_path: str = None,
                 k: int = 3,
                 s: int = 3,
                 d: int = 8,
                 beta: float = -0.9):
        torch.random.manual_seed(42)
        self.edges_path = edges_path
        self.embeddings_path = embeddings_path
        self.k = k
        self.s = s
        self.d = d

        self.adj_mat = self._load_adj_matrix()
        print("Adjacency matrix loaded")

        self.A = self._calculate_Dinv().matmul(self.adj_mat)
        print("Transition matrix constructed")

        self.L = torch.pow(self._calculate_L_matrix(), beta)
        print("L matrix constructed")

        if embeddings_path:
            pass
            self.R = self._load_embeddings_matrix()
            self.d = self.R.shape[1]
            print("Starting embeddings loaded from " + embeddings_path)
            print("Embedding dimension: " + str(self.d))

        else:
            self.R = self._random_projection()
            print("R matrix constructed")

    def _load_embeddings_matrix(self):
        m = torch.load(self.embeddings_path).float()
        return m

    def _calculate_Dinv(self) -> torch.sparse.Tensor:
        diags = self.calculate_node_degrees(self.adj_mat)
        return torch.sparse.spdiags(1/diags, torch.LongTensor([0]), self.adj_mat.shape)

    def _calculate_L_matrix(self) -> torch.sparse.Tensor:
        diags = self.calculate_node_degrees(self.adj_mat) / (2 * self.adj_mat.shape[0])
        return torch.sparse.spdiags(diags, torch.LongTensor([0]), self.adj_mat.shape)

    def _load_adj_matrix(self) -> torch.sparse.Tensor:
        edges: torch.Tensor = torch.load(self.edges_path)
        edges_other_direction = torch.index_select(edges, 1, torch.LongTensor([1, 0]))

        num_vertices = edges.max() + 1
        return torch.sparse_coo_tensor(torch.cat((edges, edges_other_direction)).t(), [1.0 for _ in range(edges.shape[0]*2)],
                                       (num_vertices, num_vertices))

    @staticmethod
    def calculate_node_degrees(matrix: torch.sparse.Tensor) -> torch.sparse.Tensor:
        return matrix.sum(dim=0).to_dense()

    def run(self) -> List[Tensor]:
        _Ns: List[Tensor] = []
        N_i: Tensor = torch.nn.functional.normalize(self.A.matmul(self.L.matmul(self.R)).to_dense())
        print("N1 done")
        _Ns.append(N_i.detach().clone())
        for i in range(self.k-1):
            N_i = torch.nn.functional.normalize(self.A.matmul(N_i))
            _Ns.append(N_i.detach().clone())
            print("N"+str(i+2) + " done")

        return _Ns

    def _random_projection(self) -> torch.Tensor:
        one_over_2_s = 1 / (2 * self.s)
        sqrt_s = math.sqrt(self.s)
        projection_values = []
        for _ in range(self.adj_mat.shape[0] * self.d):
            r = random.random()
            if r < one_over_2_s:
                projection_values.append(sqrt_s)
            elif r > (1 - one_over_2_s):
                projection_values.append(-sqrt_s)
            else:
                projection_values.append(0)

        return torch.tensor(projection_values).reshape((self.adj_mat.shape[0], self.d)).to_sparse_coo()
