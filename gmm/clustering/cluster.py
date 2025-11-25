import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from pipelineio.io_utils import save_df
from typing import Any, Literal

def label_and_score(X: pd.DataFrame, 
                    ks: tuple[int, ...] = (2, 4, 6), 
                    save: bool = True) -> tuple[dict[int, dict[Any, float]], pd.DataFrame]:
    """
    Labels each data point and calculates a Silhouette score per k-cluster.

    Parameters
    ----------
        X : DataFrame
            The DataFrame containing question responses.
        ks : tuple[int, ...]
            One or values to use as the number of clusters.
        save : bool
            Set to `True` to save the DataFrame to a CSV file. Default is `False`.

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
        gmm = GaussianMixture(n_components=k, random_state=42)
        labels = gmm.fit_predict(X)
        score = silhouette_score(X, labels)
        results[k] = dict(labels=labels, sil=score)

        df_labels = X.copy()
        df_labels["Cluster"] = labels
        if save:
            save_df(df_labels, f"{k}_clusters_labels.csv")

    # save summary df
    summary_rows = [dict(k=k, sil=results[k]["sil"]) for k in ks]
    summary = pd.DataFrame(summary_rows)
    if save:
        save_df(summary, f"sil_score_summary.csv")

    return results, summary
