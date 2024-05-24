import xgboost as xgb
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd


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


def plot_k_and_scores(k: list[int], scores: list[float], save_path: str, model_id: str):
    """
    Plot the scores for different values of k.
    :param k: list of k values
    :param scores: list of scores corresponding to the k values
    """
    plt.figure(figsize=(8, 6))  # Increase figure size

    plt.plot(k, scores, marker='o')
    plt.xlabel('k', fontsize=12, labelpad=12)  # Increase font size and set labelpad for x-axis label
    plt.ylabel('NDCG@5', fontsize=12, labelpad=12)  # Increase font size and set labelpad for y-axis label

    # Set y-axis ticks dynamically
    max_score = max(scores)
    min_score = min(scores)
    num_ticks = min(max_score - min_score + 1, 10)  # Maximum 10 ticks
    plt.locator_params(axis='y', nbins=num_ticks)
    plt.xticks(k)

    # Adjust layout to ensure all labels are visible
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        os.path.join(save_path, f"k_plot_{model_id}.png"))


def plot_n_trees_and_scores(n_trees: list[int], scores: list[float], save_path: str):
    """
    Plot the scores for different values of n_trees.
    :param n_trees: list of n_trees values
    :param scores: list of scores corresponding to the n_trees values
    """
    plt.figure(figsize=(8, 6))  # Increase figure size

    plt.plot(n_trees, scores, marker='o')
    plt.xlabel('n_trees', fontsize=12, labelpad=12)  # Increase font size and set labelpad for x-axis label
    plt.ylabel('NDCG@5', fontsize=12, labelpad=12)  # Increase font size and set labelpad for y-axis label

    # Set y-axis ticks dynamically
    max_score = max(scores)
    min_score = min(scores)
    num_ticks = min(max_score - min_score + 1, 10)  # Maximum 10 ticks
    plt.locator_params(axis='y', nbins=num_ticks)
    plt.xticks(n_trees)

    # Adjust layout to ensure all labels are visible
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        os.path.join(save_path, f"n_trees_plot_annoy.png"))


def plot_training_history(train_csv_files: list[str], series_names: list[str], save_path: str):
    if len(train_csv_files) != len(series_names):
        raise ValueError(
            "The number of CSV files must match the number of series names.")

    plt.figure(figsize=(10, 6))

    for file, name in zip(train_csv_files, series_names):
        data = pd.read_csv(file)
        iterations = data['iteration']
        ndcg_scores = data['ndcg_score']

        plt.plot(iterations, ndcg_scores, label=name)

    plt.xlabel('Iteration', fontsize=12, labelpad=12)
    plt.ylabel('NDCG', fontsize=12, labelpad=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(save_path, f"training_histories.png"))


def plot_multiple_bar_charts_position_numbers(position_csv_files: list[str],
                                              series_names: list[str], save_path: str):
    if len(position_csv_files) != len(series_names):
        raise ValueError(
            "The number of CSV files must match the number of series names.")

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    # Store the maximum y-value for both clicked and booked counts
    max_clicked = 0
    max_booked = 0

    for idx, (file, name) in enumerate(zip(position_csv_files, series_names)):
        data = pd.read_csv(file)

        # Handle the case where 'booked' column might be missing
        if 'booked' not in data.columns:
            data['booked'] = None

        clicked_counts = data['clicked'].value_counts().sort_index()
        booked_counts = data['booked'].dropna().value_counts().sort_index()

        # Plot the 'booked' column bar chart in the first subplot
        axes[0].bar(booked_counts.index + idx * 0.2, booked_counts.values, width=0.2,
                    label=name)

        # Plot the 'clicked' column bar chart in the second subplot
        axes[1].bar(clicked_counts.index + idx * 0.2, clicked_counts.values, width=0.2,
                    label=name)

        # Update the maximum y-values
        max_clicked = max(max_clicked, clicked_counts.max())
        max_booked = max(max_booked, booked_counts.max())

    # Set the y-axis limits to be the same for both subplots, with some white space
    max_y_value = max(max_clicked, max_booked)
    buffer = 0.1 * max_y_value  # Adding 10% buffer space
    axes[0].set_ylim(0, max_y_value + buffer)
    axes[1].set_ylim(0, max_y_value + buffer)

    # Configure the first subplot for 'booked'
    axes[0].set_xlabel('Position', fontsize=12, labelpad=12)
    axes[0].set_title('Booked Hotels')
    axes[0].legend()
    axes[0].grid(True)

    # Configure the second subplot for 'clicked'
    axes[1].set_xlabel('Position', fontsize=12, labelpad=12)
    axes[1].set_title('Clicked Hotels')
    axes[1].legend()
    axes[1].grid(True)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"position_distributions.png"))
    plt.close()



if __name__=="__main__":
    best_n_tree = 10
    best_k = 5
    train_csv_files = [ "plots/positions_engineer_model.csv", "plots/positions_basic_model.csv", f"plots/positions_n_trees_{best_n_tree}_k_{best_k}.csv"]
    series_names = ["Engineered LambdaMART", "Basic LambdaMART", "ANNoy"]
    save_path = "plots"
    plot_multiple_bar_charts_position_numbers(position_csv_files=train_csv_files, series_names=series_names, save_path=save_path)

    train_csv_files = [ "validation_scores/training_history_engineer_model.csv", "validation_scores/training_history_basic_model.csv"]
    series_names = ["Engineered LambdaMART", "Basic LambdaMART" ]
    save_path = "plots"
    plot_training_history(train_csv_files=train_csv_files, series_names=series_names, save_path=save_path)
