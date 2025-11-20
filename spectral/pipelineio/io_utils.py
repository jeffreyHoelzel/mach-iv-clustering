from pathlib import Path
import pandas as pd
from matplotlib.figure import Figure
from datetime import datetime
from typing import Any

ARTIFACTS_DIR = Path(f"artifacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

def ensure_dir_exists(*parts: tuple[Any, ...]) -> Path:
    path = ARTIFACTS_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_df(df: pd.DataFrame, *parts: tuple[Any, ...], fmt: str = "csv") -> Path:
    out_dir = ensure_dir_exists("data")
    out_path = out_dir.joinpath(*parts)
    if fmt == "csv":
        df.to_csv(out_path, index=False)
    elif fmt == "parquet":
        df.to_parquet(out_path, index=False)
    return out_path

def save_fig(fig: Figure, *parts: tuple[Any, ...], dpi: int = 300) -> Path:
    *folder_parts, file_name = parts
    out_dir = ensure_dir_exists(*folder_parts)
    out_path = out_dir / file_name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path