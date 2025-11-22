from pathlib import Path
import pandas as pd
from matplotlib.figure import Figure
from datetime import datetime
from typing import Any

ARTIFACTS_DIR = Path(f"artifacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

def ensure_dir_exists(*parts: tuple[Any, ...]) -> Path:
    """
    Ensures a given tuple of directory names exist and if not, creates them.

    Parameters
    ----------
        parts : tuple[Any, ...]
            A tuple of the directory/file names that may or may not exist.

    Returns
    -------
        Path
            The path to the directory where the subdirectories are located.

    Usage
    -----
    >>> ensure_dir_exists("models")
    path/to/models/
    """
    path = ARTIFACTS_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_df(df: pd.DataFrame, *parts: tuple[Any, ...], fmt: str = "csv") -> Path:
    """
    Saves a DataFrame to either a CSV or parquet output file and returns the path.

    Parameters
    ----------
        df : DataFrame
            Input DataFrame to save to a file.
        parts : tuple[Any, ...]
            A tuple of the directory/file names that may or may not exist.
        fmt : str
            The specified file format to save the DataFrame as. Default is `"csv"`, but `"parquet"`
            is also an option.

    Returns
    -------
        Path
            The output path to the new file.

    Usage
    -----
    >>> save_df(X, "X_input.csv")
    path/to/saved/X_input.csv

    >>> save_df(X, "X_input.parquet")
    path/to/saved/X_input.parquet
    """
    out_dir = ensure_dir_exists("data")
    out_path = out_dir.joinpath(*parts)
    if fmt == "csv":
        df.to_csv(out_path, index=True)
    elif fmt == "parquet":
        df.to_parquet(out_path, index=False)
    return out_path

def save_fig(fig: Figure, *parts: tuple[Any, ...], dpi: int = 300) -> Path:
    """
    Saves figure to disk as an image and returns the path.

    Parameters
    ----------
        fig : Figure
            Input figure to save to disk.
        parts : tuple[Any, ...]
            A tuple of the directory/file names that may or may not exist.
        dpi : int
            The figure's resolution in dots per inch. Default is `300`.

    Returns
    -------
        Path
            The output path to the new image.

    Usage
    -----
    >>> save_fig(fig1, "dendrograms", "ward.png")
    path/to/saved/dendrograms/ward.png
    """
    *folder_parts, filename = parts
    out_dir = ensure_dir_exists(*folder_parts)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path
