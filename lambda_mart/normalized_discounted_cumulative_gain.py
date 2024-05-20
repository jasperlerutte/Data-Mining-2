import numpy as np


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



if __name__=="__main__":
    # Example usage:
    preds = [0.8, 0.5, 0.6, 0.9, 0.4, 0.7, 0.3, 0.2, 0.1, 0.5]  # Example predicted relevance scores for 10 hotels
    labels = [1, 2, 0, 1, 0, 2, 1, 0, 0, 1]  # Example labels for 10 hotels
    ndcg_at_5 = ndcg_weighted(preds, labels, k=5)
    print("NDCG@5:", ndcg_at_5)
