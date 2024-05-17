import itertools
import os.path
import uuid
import csv
import xgboost as xgb
import dotenv
from utils.load_data import load_train_val_test_split, load_competition_data
from lambda_mart.normalized_discounted_cumulative_gain import ndcg_weighted, get_positions_target
from ray import tune, train
from ray.tune.search.hyperopt import HyperOptSearch
from lambda_mart.plot_results import plot_feature_importance_xgboost_ranker, plot_barchart_positions


xgb.config_context(verbosity=2, use_rmm=True)
YOUR_TIME_BUDGET_IN_HOURS = 0.005
YOUR_TEST_FILE = os.getenv("TEST_FILE")
YOUR_TRAIN_FILE = os.getenv("TRAIN_FILE")

config = {
    "eta": tune.loguniform(0.01, 0.2),
    "max_depth": tune.randint(3, 10),
    "min_child_weight": tune.randint(1, 50),
    "subsample": tune.uniform(0.1, 1.0),
    "colsample_bytree": tune.uniform(0.5, 1.0),
    "n_estimators": tune.randint(10, 200),
    "reg_lambda": tune.loguniform(1e-8, 1.0),
    "reg_alpha": tune.loguniform(1e-8, 1.0),
}


def train_xgb_ranker(config):
    train_data, val_data, test_data = load_train_val_test_split(
        YOUR_TRAIN_FILE,
        drop_original_targets=True, seed=420, n_rows=10000)

    y_train = train_data["label"]
    X_train = train_data.drop(columns=["label"])

    y_val = val_data["label"]
    X_val = val_data.drop(columns=["label"])

    y_test = test_data["label"]
    X_test = test_data.drop(columns=["label"])

    ranker = xgb.XGBRanker(
        tree_method="hist",
        lambdarank_num_pair_per_sample=5,
        objective="rank:ndcg",
        lambdarank_pair_method="topk",
        eval_metric="ndcg",
        subsample=config["subsample"],
        eta=config["eta"],
        max_depth=config["max_depth"],
        min_child_weight=config["min_child_weight"],
        colsample_bytree=config["colsample_bytree"],
        n_estimators=config["n_estimators"],
        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],
        early_stopping_rounds=5,
    )

    ranker.fit(X=X_train, y=y_train, verbose=True, eval_set=[(X_val, y_val)])

    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_id = str(uuid.uuid4())
    model_path = os.path.join(current_dir, "model_checkpoints", model_id + ".json")
    ranker.save_model(model_path)
    training_history = ranker.evals_result()

    # Predict on test set
    scores = ranker.predict(X_test)
    X_test["score"] = scores

    ndcg_score = 0
    n_queries = 0
    for qid in X_test['qid'].unique():
        qid_rows = X_test[X_test['qid'] == qid]
        qid_scores = qid_rows['score']
        qid_labels = y_test[qid_rows.index]
        ndcg_score += ndcg_weighted(qid_scores, qid_labels)
        n_queries += 1

    ndcg_score /= n_queries

    print(f"Model {model_id} achieved NDCG score of {ndcg_score} on the test set.")

    train.report(metrics={"ndcg": ndcg_score, "model_path": model_path,
                          "training_history": training_history, "model_id": model_id})


hyperopt = HyperOptSearch(metric="ndcg", mode="max")


analysis = tune.run(
    train_xgb_ranker,
    config=config,
    num_samples=-1,
    resources_per_trial={"cpu": 8, "gpu": 1},
    progress_reporter=tune.CLIReporter(metric_columns=["score"]),
    trial_dirname_creator=lambda trial: "tune_trial_{}".format(trial.trial_id),
    time_budget_s=3600*YOUR_TIME_BUDGET_IN_HOURS,
    search_alg=hyperopt,
)


current_dir = os.path.dirname(os.path.realpath(__file__))

best_trial = analysis.get_best_trial(metric="ndcg", mode="max")
best_model_path = best_trial.last_result["model_path"]
best_model = xgb.XGBRanker()
best_model.load_model(best_model_path)

# Get the best hyperparameters and save them
print("Best hyperparameters found were: ", analysis.get_best_config("ndcg", "max"))
with open(os.path.join(current_dir, "model_checkpoints",
    f"hyperparameters_{best_trial.last_result['model_id']}.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Hyperparameter", "Value"])
    for key, value in analysis.get_best_config("ndcg", "max").items():
        writer.writerow([key, value])

# save training history to a file
training_history = best_trial.last_result["training_history"]
with open(os.path.join(current_dir, "validation_scores",
                       f"training_history_{best_trial.last_result['model_id']}.csv"),
          "w", newline='') as f:
    list_history = training_history["validation_0"]['ndcg']
    writer = csv.writer(f)
    writer.writerow(["iteration", "ndcg_score"])
    for i in range(len(list_history)):
        writer.writerow([i, list_history[i]])

# load our own holdout set again and get predictions
_, _, test_data = load_train_val_test_split(YOUR_TRAIN_FILE,
                                            drop_original_targets=True, seed=420,
                                            n_rows=10000)
y_test = test_data["label"]
X_test = test_data.drop(columns=["label"])
scores = best_model.predict(X_test)
X_test["score"] = scores

print(f"Running predictions on own test set.")
ndcg_score = 0
n_queries = 0
booked_positions = []
clicked_positions = []
for qid in X_test['qid'].unique():
    qid_rows = X_test[X_test['qid'] == qid]
    qid_scores = qid_rows['score']
    qid_labels = y_test[qid_rows.index]
    ndcg_score += ndcg_weighted(qid_scores, qid_labels)
    booked_positions.extend(get_positions_target(scores=qid_scores, y_true=qid_labels, target_label=5))
    clicked_positions.extend(get_positions_target(scores=qid_scores, y_true=qid_labels, target_label=1))
    n_queries += 1

ndcg_score /= n_queries

print(f"NDCG score on own test set of {n_queries} queries: ", ndcg_score)

# save ndcg score to a file
with open(os.path.join(current_dir, "validation_scores",
                        f"ndcg_score_{best_trial.last_result['model_id']}.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ndcg', 'queries'])
    writer.writerow([ndcg_score, n_queries])

# save predicted positions of the clicked and booked hotels in our test set
with open(os.path.join(current_dir, "plots",
    f"positions_{best_trial.last_result['model_id']}.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['clicked', 'booked'])
    for clicked, booked in itertools.zip_longest(clicked_positions, booked_positions):
        writer.writerow([clicked, booked])

plot_path = os.path.join(current_dir, "plots")
plot_barchart_positions(positions=booked_positions, model_id=best_trial.last_result["model_id"], target="booked", save_path=plot_path)
plot_barchart_positions(positions=clicked_positions, model_id=best_trial.last_result["model_id"], target="clicked", save_path=plot_path)
plot_feature_importance_xgboost_ranker(model=best_model, model_id=best_trial.last_result["model_id"],
                                       k=10, save_feature_importance=True, save_path=plot_path)

# load competition test set
print("Running predictions on competition test set...")
competition_test_data = load_competition_data(YOUR_TEST_FILE, n_rows=10000)
scores = best_model.predict(competition_test_data)
competition_test_data["score"] = scores

with open(os.path.join(current_dir, "predictions",
                       f"predictions_{best_trial.last_result['model_id']}.csv"), "w", newline='') as f:
    f.write("srch_id,prop_id\n")
    for qid in competition_test_data['qid'].unique():
        qid_rows = competition_test_data[competition_test_data['qid'] == qid]
        qid_scores = qid_rows['score']
        qid_rows = qid_rows.sort_values(by='score', ascending=False)

        for index, row in qid_rows.iterrows():
            f.write(f"{qid},{int(row['prop_id'])}\n")
