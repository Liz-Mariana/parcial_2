from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_dataset(name):
    """Carga data/<name>.csv y devuelve (x, y) como arreglos de numpy."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return df["x"].to_numpy(), df["y"].to_numpy()