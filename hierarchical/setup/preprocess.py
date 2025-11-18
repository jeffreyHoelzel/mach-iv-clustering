import pandas as pd
from .config import DATA_PATH, QUESTION_COLS, RANDOM_STATE, SAMPLE_N
from io.io_utils import save_df

def load_raw() -> pd.DataFrame:
    """Loads the complete CSV from disk into a DataFrame and returns for machine learning use."""
    return pd.read_csv(DATA_PATH)

def prep_sample(save: bool = False):
    """
    Prepares the machine learning input data with N rows and optionally saves to output file.

    Parameters
    ----------
        save : bool
            Set to `True` to save the DataFrame to a CSV file. Default is `False`.

    Returns
    -------
        DataFrame
            A DataFrame of size Nx20 ready for clustering.
    """
    df = load_raw()
    X = df[QUESTION_COLS].copy()
    X_clean = X.dropna().astype(int)
    X_sample = X_clean.sample(n=SAMPLE_N, random_state=RANDOM_STATE)
    save_df(X_sample, "Xs_sample.csv")
    return X_sample
