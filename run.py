
import os
import argparse
import json
import time

import numpy as np
import pandas as pd
import scanpy as sc

# Read arguments supplied by OmniBenchmark
parser = argparse.ArgumentParser(
    description="Random clustering baseline"
)

parser.add_argument("--output_dir", "-o", required=True)
parser.add_argument("--name", "-n", required=True)
parser.add_argument("--data.h5ad", dest="data_h5ad", required=True)
parser.add_argument(
    "--data.clusters_truth_num",
    dest="clusters_truth_num_path",
    required=True
)
parser.add_argument("--seed", type=int, default=42)

args, _ = parser.parse_known_args()

os.makedirs(args.output_dir, exist_ok=True)

# Load the dataset
adata = sc.read_h5ad(args.data_h5ad)
adata.var_names_make_unique()

# Read the number of true clusters
with open(args.clusters_truth_num_path) as f:
    n_clusters = int(f.read().strip())

if n_clusters < 1:
    raise ValueError("Number of clusters must be positive")

# Generate reproducible random cluster assignments
start = time.perf_counter()

rng = np.random.default_rng(args.seed)

labels = rng.integers(
    low=0,
    high=n_clusters,
    size=adata.n_obs
).astype(str)

elapsed = time.perf_counter() - start

# Save assignments using the same format as Scanpy
clusters = pd.DataFrame({
    "cell_id": adata.obs_names,
    "louvain": labels,
    "leiden": labels
})

clusters.to_csv(
    os.path.join(
        args.output_dir,
        f"{args.name}.clusters.tsv"
    ),
    sep="\t",
    index=False
)

# Save timing information
timings = {
    "gpu_load": None,
    "find_mit_gene": None,
    "filter": None,
    "normalization": None,
    "hvg": None,
    "scaling": None,
    "pca": None,
    "t_sne": None,
    "umap": None,
    "louvain": elapsed,
    "leiden": elapsed
}

with open(
    os.path.join(args.output_dir, f"{args.name}.timings.json"),
    "w"
) as f:
    json.dump(timings, f, indent=2)

# Save clustering metadata
clustering_info = {
    "resolutions": {
        "louvain": None,
        "leiden": None
    },
    "num_runs": {
        "louvain": 1,
        "leiden": 1
    }
}

with open(
    os.path.join(
        args.output_dir,
        f"{args.name}.clustering_info.json"
    ),
    "w"
) as f:
    json.dump(clustering_info, f, indent=2)

# Generate PCA coordinates for the metrics stage
# This is separate from the random clustering itself.
n_components = min(
    30,
    adata.n_obs - 1,
    adata.n_vars - 1
)

if n_components < 1:
    raise ValueError("Dataset is too small for PCA")

sc.pp.pca(adata, n_comps=n_components)

pca_df = pd.DataFrame(
    adata.obsm["X_pca"],
    index=adata.obs_names,
    columns=[
        f"PC{i + 1}"
        for i in range(n_components)
    ]
)

pca_df.to_csv(
    os.path.join(
        args.output_dir,
        f"{args.name}.pca.tsv"
    ),
    sep="\t",
    index=True,
    index_label="cell_id"
)

print(
    f"Random clustering complete: "
    f"{adata.n_obs} cells, "
    f"{n_clusters} possible clusters, "
    f"seed={args.seed}"
)
