import numpy as np
import pandas as pd

def random_clustering(cell_ids, n_clusters, seed=42):
    rng = np.random.default_rng(seed)

    labels = rng.integers(
        0,
        n_clusters,
        size=len(cell_ids)
    )

    return pd.DataFrame({
        "cell_id": cell_ids,
        "louvain": labels.astype(str),
        "leiden": labels.astype(str)
    })
