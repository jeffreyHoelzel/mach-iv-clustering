import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score
from io.io_utils import save_df
from typing import Any, LiteralString

def label_and_score(X: pd.DataFrame, 
                    Z: np.ndarray, 
                    ks: tuple[int, ...] = (2, 3, 4), 
                    save: bool = True, 
                    linkage: LiteralString = "single" | "complete" | "average" | "ward") -> tuple[dict[int, dict[Any, float]], pd.DataFrame]:
    """
    Labels each data point and calculates a Silhouette score per k-cluster.

    Parameters
    ----------
        X : DataFrame
            The DataFrame containing question responses.
        Z : NDArray
            The linkage to use for clustering.
        ks : tuple[int, ...]
            One or values to use as the number of clusters.
        save : bool
            Set to `True` to save the DataFrame to a CSV file. Default is `False`.
        linkage : LiteralString
            The type of linkage used for Z.

    Returns
    -------
        results : dict[int, dict[Any, float]
            A dictionary mapping cluster sizes to a dictionary of labels and Silhouette scores.
        summary : DataFrame
            A summary DataFrame of the cluster size and Silhouette scores.
    """
    # output dict and/or df
    results = {}
    for k in ks:
        labels = fcluster(Z, k, criterion="maxclust")
        score = silhouette_score(X, labels)
        results[k] = dict(labels=labels, sil=score)

        df_labels = X.copy()
        df_labels["Cluster"] = labels
        if save:
            save_df(df_labels, f"{linkage}_{k}_clusters_labels.csv")

    # save summary df
    summary_rows = [dict(k=k, sil=results[k]["sil"]) for k in ks]
    summary = pd.DataFrame(summary_rows)
    if save:
        save_df(summary, f"{linkage}_sil_score_summary.csv")

    return results, summary
