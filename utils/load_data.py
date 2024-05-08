import os
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt


def load_data(data_file: str, n_rows: int = None) -> pd.DataFrame:
    load_dotenv()
    folder = os.getenv("DATA_FOLDER")
    data_path = os.path.join(folder, data_file)
    if n_rows is not None:
        df = pd.read_csv(data_path, nrows=n_rows)
    else:
        df = pd.read_csv(data_path)
    return df
