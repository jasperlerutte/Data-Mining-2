import os.path

from numba import cuda
import numpy as np
import uuid
import csv
import xgboost as xgb
from utils.load_data import load_train_val_test_split
from lambda_mart.normalized_discounted_cumulative_gain import ndcg_weighted
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
from sklearn.metrics import ndcg_score



xgb.config_context(verbosity=2, use_rmm=True)

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
    train_data, val_data, test_data = load_train_val_test_split("training_set_VU_DM.csv", n_rows=1000,
                                                 drop_original_targets=True)

    y_train = train_data["label"]
    X_train = train_data.drop(columns=["label", "position"])

    y_val = val_data["label"]
    X_val = val_data.drop(columns=["label", "position"])

    y_test = test_data["label"]
    X_test = test_data.drop(columns=["label", "position"])

    device = "gpu" if cuda.is_available() else "cpu"

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
        early_stopping_rounds=10
    )

    ranker.fit(X=X_train, y=y_train, verbose=True, eval_set=[(X_val, y_val)])


    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_id = str(uuid.uuid4())
    model_path = os.path.join(current_dir, "model_checkpoints", model_id + ".json")
    ranker.save_model(model_path)

    training_history = ranker.evals_result()

    scores = ranker.predict(X_test)
    sorted_idx = np.argsort(scores)[::-1]
    scores = scores[sorted_idx]

    print(scores)
    train.report(metrics={"ndcg": scores[0], "model_path": model_path, "training_history": training_history, "model_id": model_id})


asha_sched = ASHAScheduler(metric="ndcg", mode="max")

analysis = tune.run(
    train_xgb_ranker,
    config=config,
    num_samples=2,
    scheduler=asha_sched,
    resources_per_trial={"cpu": 8, "gpu": 1},
    progress_reporter=tune.CLIReporter(metric_columns=["score"]),
    trial_dirname_creator=lambda trial: "tune_trial_{}".format(trial.trial_id),
)

print("Best hyperparameters found were: ", analysis.get_best_config("ndcg", "max"))

best_trial = analysis.get_best_trial(metric="ndcg", mode="max")
best_model_path = best_trial.last_result["model_path"]
best_model = xgb.XGBRanker()
best_model.load_model(best_model_path)
training_history = best_trial.last_result["training_history"]

# save training history to a file
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "validation_scores", f"training_history_{best_trial.last_result['model_id']}.csv"), "w") as f:
    list_history = training_history["validation_0"]['ndcg']
    writer = csv.writer(f)
    writer.writerow(list_history)


