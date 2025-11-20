from pathlib import Path

DATA_PATH: Path = Path("../data/MACH_data/data.cleaned.csv")
QUESTION_COLS: list[str] = [f"Q{i}A" for i in range(1, 21)]
RANDOM_STATE: int = 42
SAMPLE_N: int = 5000