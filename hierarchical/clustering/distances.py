import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from io.io_utils import ensure_dir_exists
import joblib

def compute_distances(X: pd.DataFrame, save: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Euclidean and Manhattan distance matrices and optionally saves to .npy files.

    Parameters
    ----------
        X : DataFrame or array-like
            The input array for calculating distance matrices.
        save : bool
            Set to `True` to save the DataFrame to a CSV file. Default is `False`.

    Returns
    -------
        euclidean : NDArray
            The computed Euclidean distance matrix.
        manhattan : NDArray
            The computed Manhattan distance matrix.  
    """
    euclidean = squareform(pdist(X, metric="euclidean"))
    manhattan = squareform(pdist(X, metric="cityblock"))

    if save:
        out_dir = ensure_dir_exists("distance_matrices")
        np.save(out_dir / "euclidean.npy", euclidean)
        np.save(out_dir / "manhattan.npy", manhattan)

    return euclidean, manhattan

def compute_default_linkages(X: pd.DataFrame, 
                             save: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes linkage using default input DataFrame.

    Parameters
    ----------
        X : DataFrame or array-like
            The input array for computing linkages.
        save : bool
            Set to `True` to save the linkage results to a .joblib file. Default is `False`.

    Returns
    -------
        Z_single, Z_complete, Z_average, Z_ward : four NDArrays
            The computed linkage results.
    """
    Z_single = linkage(X, method="single")
    Z_complete = linkage(X, method="complete")
    Z_average = linkage(X, method="average")
    Z_ward = linkage(X, method="ward")

    if save:
        out_dir = ensure_dir_exists("default_linkages")
        joblib.dump(dict(Z_single_default=Z_single,
                         Z_complete_default=Z_complete,
                         Z_average_default=Z_average,
                         Z_ward_default=Z_ward), 
                    out_dir / "default_linkages.joblib")
        
    return Z_single, Z_complete, Z_average, Z_ward

def compute_euclidean_linkages(X_euclidean: np.ndarray, 
                               save: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes linkage using Euclidean input NDArray.

    Parameters
    ----------
        X_euclidean : NDArray or array-like
            The precomputed pairwise Euclidean input array for computing linkages.
        save : bool
            Set to `True` to save the linkage results to a .joblib file. Default is `False`.

    Returns
    -------
        Z_single, Z_complete, Z_average : three NDArrays
            The computed linkage results using precomputed Euclidean pairwise distance matrices.
    """
    Z_single = linkage(X_euclidean, method="single")
    Z_complete = linkage(X_euclidean, method="complete")
    Z_average = linkage(X_euclidean, method="average")

    if save:
        out_dir = ensure_dir_exists("euclidean_linkages")
        joblib.dump(dict(Z_single_euclidean=Z_single,
                         Z_complete_euclidean=Z_complete,
                         Z_average_euclidean=Z_average), 
                    out_dir / "euclidean_linkages.joblib")
        
    return Z_single, Z_complete, Z_average

def compute_manhattan_linkages(X_manhattan: np.ndarray, 
                               save: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes linkage using Manhattan input NDArray.

    Parameters
    ----------
        X_manhattan : NDArray or array-like
            The precomputed pairwise Manhattan input array for computing linkages.
        save : bool
            Set to `True` to save the linkage results to a .joblib file. Default is `False`.

    Returns
    -------
        Z_single, Z_complete, Z_average : three NDArrays
            The computed linkage results using precomputed Manhattan pairwise distance matrices.
    """
    Z_single = linkage(X_manhattan, method="single")
    Z_complete = linkage(X_manhattan, method="complete")
    Z_average = linkage(X_manhattan, method="average")

    if save:
        out_dir = ensure_dir_exists("manhattan_linkages")
        joblib.dump(dict(Z_single_manhattan=Z_single,
                         Z_complete_manhattan=Z_complete,
                         Z_average_manhattan=Z_average), 
                    out_dir / "manhattan_linkages.joblib")
        
    return Z_single, Z_complete, Z_average

