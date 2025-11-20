import numpy as np
import pandas as pd
from typing import Any

from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

from pipelineio.io_utils import save_df
from setup.config import RANDOM_STATE


def _build_knn_rbf_similarity(
    X: pd.DataFrame,
    n_neighbors: int = 15
):
    """
    Build a k-NN graph and convert distances to RBF similarities.
    Returns a sparse similarity matrix.
    """
    # k-NN distance graph (sparse)
    W = kneighbors_graph(
        X,
        n_neighbors=n_neighbors,
        mode="distance",
        include_self=False,
    )

    W = 0.5 * (W + W.T)

    sigma = np.median(W.data)
    W_sim = W.copy()
    W_sim.data = np.exp(-(W_sim.data ** 2) / (2 * sigma ** 2))

    return W_sim


def _compute_spectral_embedding(
    X: pd.DataFrame,
    n_components: int = 3,
    n_neighbors: int = 15,
):
    """
    Compute spectral embedding using the normalized graph Laplacian.
    Returns an (n_samples, n_components) array.
    """
    W_sim = _build_knn_rbf_similarity(X, n_neighbors=n_neighbors)

    L = csgraph.laplacian(W_sim, normed=True)

    vals, vecs = eigsh(L, k=n_components + 1, which="SM")
    embedding = vecs[:, 1 : n_components + 1]
    return embedding


def label_and_score(
    X: pd.DataFrame,
    ks: tuple[int, ...] = (2, 3, 4),
    save: bool = True,
    prefix: str = "spectral",
    n_neighbors: int = 15,
) -> tuple[dict[int, dict[str, Any]], pd.DataFrame]:
    """
    Run spectral clustering for the given k values, compute
    silhouette scores, and optionally save labels and summary.

    Parameters
    ----------
    X : DataFrame
        Input features (MACH item responses).
    ks : tuple[int, ...]
        Cluster counts (e.g., (2, 3, 4)).
    save : bool
        If True, save labels and summary CSVs.
    prefix : str
        Prefix used when naming output files.
    n_neighbors : int
        Number of neighbors for k-NN graph.

    Returns
    -------
    results : dict
        results[k]["labels"] and results[k]["sil"]
    summary : DataFrame
        rows = (k, sil)
    """
    n_components = max(ks)
    embedding = _compute_spectral_embedding(
        X,
        n_components=n_components,
        n_neighbors=n_neighbors,
    )

    results: dict[int, dict[str, Any]] = {}

    results["embedding"] = embedding

    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = km.fit_predict(embedding[:, :n_components])

        sil = silhouette_score(X, labels)
        results[k] = {"labels": labels, "sil": sil}

        if save:
            df_labels = X.copy()
            df_labels["Cluster"] = labels
            save_df(df_labels, f"{prefix}_{k}_clusters_labels.csv")

    summary_rows = [dict(k=k, sil=results[k]["sil"]) for k in ks]
    summary = pd.DataFrame(summary_rows)
    if save:
        save_df(summary, f"{prefix}_sil_score_summary.csv")

    return results, summary
