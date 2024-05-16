import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from utils.create_targets import add_target_column


def load_data(data_file: str, n_rows: int = None) -> pd.DataFrame:
    load_dotenv()
    folder = os.getenv("DATA_FOLDER")
    data_path = os.path.join(folder, data_file)
    if n_rows is not None:
        df = pd.read_csv(data_path, nrows=n_rows)
    else:
        df = pd.read_csv(data_path)
    return df


def load_competition_data(data_file: str, n_rows: int = None) -> pd.DataFrame:
    load_dotenv()
    folder = os.getenv("DATA_FOLDER")
    data_path = os.path.join(folder, data_file)
    if n_rows is not None:
        df = pd.read_csv(data_path, nrows=n_rows)
    else:
        df = pd.read_csv(data_path)
    df.rename(columns={"srch_id": "qid"}, inplace=True)
    df = df.sort_values(by="qid")

    if "date_time" in df.columns:
        df.drop(columns=["date_time"], inplace=True)
    return df


def load_train_val_test_split(data_file: str, n_rows: int = None, frac_val: float = 0.2,
                              frac_test: float = 0.2,
                              drop_original_targets: bool = True,
                              booking_score: int = 5, clicking_score: int = 1,
                              seed: int = 420) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    df = load_data(data_file, n_rows=n_rows)

    columns_to_drop = ["date_time", "gross_bookings_usd", "position"]

    for column in columns_to_drop:
        if column in df.columns:
            df.drop(columns=[column], inplace=True)

    # `Create targets`
    df = add_target_column(df, booking_score=booking_score,
                           clicking_score=clicking_score)

    if drop_original_targets:
        df = df.drop(columns=["booking_bool", "click_bool"])

    # Rename the srch_id column to qid
    df.rename(columns={"srch_id": "qid"}, inplace=True)

    # sort dataframe by qid
    df = df.sort_values(by="qid")

    # randomly split in test and training set based on qid
    train_size = int((1 - frac_test) * df["qid"].nunique())
    train_qids = np.random.choice(df["qid"].unique(), train_size, replace=False)
    train = df[df["qid"].isin(train_qids)]
    test = df[~df["qid"].isin(train_qids)]

    # split training set into train and validation set
    train_size = int((1 - frac_val) * train["qid"].nunique())
    train_qids = np.random.choice(train["qid"].unique(), train_size,
                                  replace=False)
    val = train[~train["qid"].isin(train_qids)]
    train = train[train["qid"].isin(train_qids)]
    return train, val, test
