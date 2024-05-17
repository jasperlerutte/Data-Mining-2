import itertools
import os.path
import sys
import uuid
import csv
import xgboost as xgb
# from utils.load_data import load_train_val_test_split, load_competition_data
# from lambda_mart.normalized_discounted_cumulative_gain import ndcg_weighted, get_positions_target

from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
# from lambda_mart.plot_results import plot_feature_importance_xgboost_ranker, plot_barchart_positions

import pandas as pd
import numpy as np


# Add the parent directory to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


import xgboost as xgb
import os
import csv
import matplotlib.pyplot as plt


def plot_feature_importance_xgboost_ranker(model: xgb.XGBRanker, model_id: str,
                                           save_path: str, k=10,
                                           save_feature_importance=True):
    """
    Plot feature importance graph for XGBoost Ranker model.

    Args:
    - model (xgb.XGBRanker): XGBoost Ranker model object.
    - k (int): Number of top features to consider. Default is 10.
    """
    # Get the underlying booster and features
    booster = model.get_booster()
    importance_scores = booster.get_fscore()
    sorted_scores = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

    top_k_features = [x[0] for x in sorted_scores[:k]]
    top_k_scores = [x[1] for x in sorted_scores[:k]]

    # Increase the width of the figure to accommodate longer feature names
    plt.figure(figsize=(15, 8))

    # Plot feature importance
    plt.barh(range(len(top_k_features)), top_k_scores, align='center')
    plt.yticks(range(len(top_k_features)), top_k_features)

    plt.xlabel('Feature Importance Score', fontsize=14, labelpad=20)  # Increase font size and set labelpad for x-axis label
    plt.ylabel('Feature', fontsize=14, labelpad=20)  # Increase font size and set labelpad for y-axis label

    # Increase font size for tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Order bars by descending feature score
    plt.gca().invert_yaxis()

    if save_feature_importance:
        plt.tight_layout()  # Adjust layout to ensure all labels are visible
        plt.savefig(os.path.join(save_path, f"feature_importance_{model_id}.png"))
        with open(os.path.join(save_path, f"feature_importance_{model_id}.csv"), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Feature", "Importance Score"])
            for feature, score in sorted_scores:
                writer.writerow([feature, score])


def plot_barchart_positions(positions: list[int], target: str, model_id: str, save_path: str):
    """
    Plot a bar chart of the integers with bars 1 - 10+
    :param model_id: id of the model we are saving positions from
    :param target: clicked or booked
    :param positions: list of positions as predicted by the model
    """
    position_counts = {i: 0 for i in range(1, 11)}
    for pos in positions:
        if pos <= 9:
            position_counts[pos] += 1
        else:
            position_counts[10] += 1

    plt.figure(figsize=(8, 6))  # Increase figure size

    plt.bar(position_counts.keys(), position_counts.values())
    plt.xlabel('Position', fontsize=12, labelpad=12)  # Increase font size and set labelpad for x-axis label

    # change label of 10 to 10+
    plt.xticks(list(position_counts.keys()), [str(i) if i < 10 else '10+' for i in position_counts.keys()])

    # Set y-axis ticks dynamically
    max_count = max(position_counts.values())
    num_ticks = min(max_count + 1, 10)  # Maximum 10 ticks
    plt.locator_params(axis='y', nbins=num_ticks)

    # Adjust layout to ensure all labels are visible
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_path, f"position_distribution_{model_id}_{target}.png"))








def add_target_column(data: pd.DataFrame, booking_score: int = 5, clicking_score: int = 1) -> pd.DataFrame:
    data['label'] = data.apply(lambda row: booking_score if row['booking_bool'] else (clicking_score if row['click_bool'] else 0), axis=1)
    return data

def load_competition_data(data_file: str, n_rows: int = None) -> pd.DataFrame:
    if n_rows is not None:
        df = pd.read_csv("C:/Users/esrio_0v2bwuf/Desktop/Master_AI/Data_Mining_Techniques/Assignments/Assignment2/Data-Mining-2/Data/test_complete.csv", nrows=n_rows)
    else:
        df = pd.read_csv("C:/Users/esrio_0v2bwuf/Desktop/Master_AI/Data_Mining_Techniques/Assignments/Assignment2/Data-Mining-2/Data/test_complete.csv")
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

    if n_rows is not None:
        df = pd.read_csv("C:/Users/esrio_0v2bwuf/Desktop/Master_AI/Data_Mining_Techniques/Assignments/Assignment2/Data-Mining-2/Data/train_complete.csv", nrows=n_rows)
    else:
        df = pd.read_csv("C:/Users/esrio_0v2bwuf/Desktop/Master_AI/Data_Mining_Techniques/Assignments/Assignment2/Data-Mining-2/Data/train_complete.csv")

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



def ndcg_weighted(preds, labels):
    k = 5
    dcg = 0.0
    ideal_dcg = 0.0

    # Combine preds and labels into a list of tuples and sort by predicted relevance (in descending order)
    sorted_data = sorted(zip(preds, labels), key=lambda x: x[0], reverse=True)

    # Keep top k data points
    sorted_data = sorted_data[:k]

    # Calculate DCG
    for i, (pred, label) in enumerate(sorted_data):
        dcg += (2 ** label - 1) / np.log2(i + 2)

    # Sort labels by relevance (in descending order) and keep top k for calculating Ideal DCG
    ideal_labels = sorted(labels, reverse=True)
    for i, label in enumerate(ideal_labels[:k]):
        ideal_dcg += (2 ** label - 1) / np.log2(i + 2)

    # Calculate NDCG
    if ideal_dcg == 0:
        ndcg = 0.0
    else:
        ndcg = dcg / ideal_dcg

    return ndcg


def get_positions_target(scores: list, y_true: list, target_label: int):
    sorted_data = sorted(zip(scores, y_true), key=lambda x: x[0], reverse=True)
    positions = []
    for i, (pred, label) in enumerate(sorted_data):
        if label == target_label:
            positions.append(i+1)
    return positions



# if __name__=="__main__":
#     # Example usage:
#     preds = [0.8, 0.5, 0.6, 0.9, 0.4, 0.7, 0.3, 0.2, 0.1, 0.5]  # Example predicted relevance scores for 10 hotels
#     labels = [1, 2, 0, 1, 0, 2, 1, 0, 0, 1]  # Example labels for 10 hotels
#     ndcg_at_5 = ndcg_weighted(preds, labels, k=5)
#     print("NDCG@5:", ndcg_at_5)



