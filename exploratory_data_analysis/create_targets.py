from utils.load_data import load_data
import pandas as pd
import numpy as np


def add_target_column(data: pd.DataFrame) -> pd.DataFrame:
    # Add a new column named 'label' based on conditions
    data['label'] = data.apply(lambda row: 2 if row['booking_bool'] else (1 if row['click_bool'] else 0), axis=1)
    return data


if __name__ == "__main__":
    data = load_data("training_set_VU_DM.csv", n_rows=1000)
    data = add_target_column(data)
    print(data.head())