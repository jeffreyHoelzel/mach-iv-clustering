import pandas as pd
from .config import DATA_PATH, QUESTION_COLS, RANDOM_STATE, SAMPLE_N
from pipelineio.io_utils import save_df

def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

def prep_sample(save: bool = False, use_all: bool = False) -> pd.DataFrame:
    df = load_raw()
    X = df[QUESTION_COLS].copy()
    X_clean = X.dropna().astype(int)
    X_sample = X_clean.sample(n=SAMPLE_N, random_state=RANDOM_STATE) if not use_all else X_clean
    if save and not use_all:
        save_df(X_sample, f"Xs_{SAMPLE_N}.csv")
    elif save and use_all:
        save_df(X_sample, "Xs_all.csv")
    return X_sample