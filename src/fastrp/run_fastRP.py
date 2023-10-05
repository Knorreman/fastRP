import argparse

import torch

import os
from _fastRP import FastRP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings_path",
        default=None,
        type=str,
        help="Path to a pyTorch tensor file containing embedding vectors"
    )
    parser.add_argument(
        "--edges_path",
        type=str,
        help="Path to a pyTorch tensor file containing edges"
    )

    parser.add_argument(
        "--beta",
        default=-0.9,
        type=str,
        help="beta param from the paper"
    )
    parser.add_argument(
        "--dims",
        default=100,
        type=str,
        help="Embedding dimensions (If not using a provided embeddings file)"
    )


    def list_of_floats(arg):
        return list(map(float, arg.split(',')))
    parser.add_argument(
        "--weights",
        default=[1.0, 1.0],
        type=list_of_floats,
        help="Iteration weights (ex = --weights 0.8,0.7)"
    )
    parser.add_argument(
        "--r0",
        default=1.0,
        type=float,
        help="Self weight (only applicable if you provide an embeddings file)"
    )
    parser.add_argument(
        "--output_path_prefix",
        required=True,
        type=str,
        help="Where to save the new embeddings file"
    )

    opts = parser.parse_args()

    edges_path: str = opts.edges_path
    embeddings_path: str = opts.embeddings_path
    print("Loading edges from " + edges_path)
    if embeddings_path:
        print("Loading embeddings from " + embeddings_path)

    weights = opts.weights
    r0 = opts.r0

    fastRP = FastRP(edges_path,
                    embeddings_path,
                    k=len(weights),
                    d=opts.dims,
                    beta=opts.beta)

    results = fastRP.run()

    d = fastRP.d
    n = fastRP.adj_mat.shape[0]

    N = torch.zeros((n, d))

    if embeddings_path:
        N += r0 * fastRP.R

    for i, _N in enumerate(results):
        N += weights[i] * _N

    print(N)

    if embeddings_path:
        file_name = os.path.split(embeddings_path)[-1].split(".")[0] + f"_{weights}"
    else:
        file_name = f"no_init_{d}_{weights}"
    full_file_path = opts.output_path_prefix + file_name + ".pt"
    print("Saving embeddings file to: " + full_file_path)
    torch.save(N, full_file_path)
