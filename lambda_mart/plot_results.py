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




