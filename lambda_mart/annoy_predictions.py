import os
import csv
import itertools
import numpy as np
from annoy import AnnoyIndex
from utils.load_data import load_train_val_test_split
from dotenv import load_dotenv
import pandas as pd
from lambda_mart.normalized_discounted_cumulative_gain import ndcg_weighted, \
    get_positions_target
from lambda_mart.plot_results import plot_barchart_positions, plot_k_and_scores, plot_n_trees_and_scores

load_dotenv()
YOUR_TEST_FILE = os.getenv("TEST_FILE")
YOUR_TRAIN_FILE = os.getenv("TRAIN_FILE")


def score_single_item(item: pd.Series, y_train: pd.Series, model: AnnoyIndex, k: int = 5, ) -> float:
    k_nn_indices = model.get_nns_by_vector(vector=item, n=k, search_k=10)
    k_nn_labels = y_train.iloc[k_nn_indices]
    return k_nn_labels.mean()


def evaluate_set(set: pd.DataFrame, y_test: pd.Series, model: AnnoyIndex, y_train: pd.Series, k: int = 5) -> dict:
    ndcg = 0
    n_queries = 0
    booked_positions = []
    clicked_positions = []
    print(f"Starting evaluation with k={k}")
    total_queries = len(set["qid"].unique())
    for qid in set["qid"].unique():
        print(f"\rQuery {n_queries}/{total_queries}", end="")
        query = set[set["qid"] == qid]
        query = query.drop(columns=["qid"])
        scores = []
        for i in range(query.shape[0]):
            score = score_single_item(item=query.iloc[i], y_train=y_train, model=model, k=k)
            scores.append(score)

        # set scores in query
        query["score"] = scores

        # calculate ndcg
        query_scores = query["score"]
        true_labels = y_test[query.index]
        ndcg += ndcg_weighted(query_scores, true_labels)
        n_queries += 1
        booked_positions.extend(
            get_positions_target(scores=query_scores, y_true=true_labels, target_label=5))
        clicked_positions.extend(
            get_positions_target(scores=query_scores, y_true=true_labels, target_label=1))

    return {"ndcg": ndcg / n_queries, "booked_positions": booked_positions, "clicked_positions": clicked_positions}


print("Loading data")
train_data, val_data, test_data = load_train_val_test_split(
    YOUR_TRAIN_FILE,
    drop_original_targets=True, seed=420)

train_data = train_data.drop(columns=["qid"])

y_train = train_data["label"]
X_train = train_data.drop(columns=["label"])

y_val = val_data["label"]
X_val = val_data.drop(columns=["label"])

y_test = test_data["label"]
X_test = test_data.drop(columns=["label"])

n_features = X_train.shape[1]
n_trees_to_test = [n for n in range(5, 11)]
possible_k = [k for k in range(1, 6)]

print("Building model")
model = AnnoyIndex(n_features, 'angular')
annoy_indices = []
for i in range(X_train.shape[0]):
    model.add_item(i, X_train.iloc[i].values)
    annoy_indices.append(i)

y_train.reindex(annoy_indices)


results = {}

for n_trees in n_trees_to_test:
    print(f"Building model with {n_trees} trees")
    model.unbuild()
    model.build(n_trees)
    k_results = {}
    for k in possible_k:
        one_res = evaluate_set(set=X_val, y_test=y_val, model=model, y_train=y_train, k=k)
        k_results[k] = one_res

    # print ndcg for best k
    best_k = max(k_results, key=lambda k: k_results[k]["ndcg"])
    print(f"n_trees: {n_trees}, best k: {best_k}, ndcg: {k_results[best_k]['ndcg']}")
    results[n_trees] = k_results

# Iterate through the nested dictionary
best_n_tree = None
best_k = None
all_scores_per_n_tree = []
max_ndcg_value = float('-inf')
for n_tree, n_tree_results in results.items():
    best_score_n_tree = float('-inf')
    for k, k_result in n_tree_results.items():
        ndcg_value = k_result.get("ndcg", float('-inf'))
        if ndcg_value > best_score_n_tree:
            best_score_n_tree = ndcg_value
        if ndcg_value > max_ndcg_value:
            max_ndcg_value = ndcg_value
            best_n_tree = n_tree
            best_k = k
    all_scores_per_n_tree.append(best_score_n_tree)


# make barcharts positions of best n_tree
best_results = results[best_n_tree]
best_k_results = best_results[best_k]

current_dir = os.path.dirname(os.path.realpath(__file__))
plot_path = os.path.join(current_dir, "plots")

all_k = []
all_scores = []
for k, result in best_results.items():
    all_k.append(k)
    all_scores.append(result["ndcg"])

plot_k_and_scores(k=all_k, scores=all_scores, model_id=f"n_trees_{best_n_tree}", save_path=plot_path)
plot_n_trees_and_scores(n_trees=n_trees_to_test, scores=all_scores_per_n_tree, save_path=plot_path)

model.unbuild()
model.build(best_n_tree)
final_results = evaluate_set(set=X_test, y_test=y_test, model=model, y_train=y_train, k=best_k)
print(f"Final results: {final_results}")

# save predicted positions of the clicked and booked hotels in our test set
with open(os.path.join(current_dir, "plots",
                       f"positions_n_trees_{best_n_tree}_k_{best_k}.csv"), "w",
          newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['clicked', 'booked'])
    for clicked, booked in itertools.zip_longest(final_results["booked_positions"], final_results["clicked_positions"]):
        writer.writerow([clicked, booked])


plot_barchart_positions(positions=final_results["booked_positions"], target="booked", model_id=f"n_trees_{best_n_tree}_k_{best_k}", save_path=plot_path)
plot_barchart_positions(positions=final_results["clicked_positions"], target="clicked", model_id=f"n_trees_{best_n_tree}_k_{best_k}", save_path=plot_path)

# save k and scores in a csv
with open(os.path.join(plot_path, f"n_trees_{best_n_tree}_k_and_scores.csv"), "w") as f:
    f.write("k,score\n")
    for k, score in zip(all_k, all_scores):
        f.write(f"{k},{score}\n")

# save n_trees and scores in a csv
with open(os.path.join(plot_path, f"n_trees_and_scores.csv"), "w") as f:
    f.write("n_trees,score\n")
    for n_tree, score in zip(n_trees_to_test, all_scores_per_n_tree):
        f.write(f"{n_tree},{score}\n")

# save final score in a csv
with open(os.path.join(current_dir, "validation_scores", f"annoy_final_score.csv"), "w") as f:
    f.write("ndcg_test\n")
    f.write(f"{final_results['ndcg']}\n")