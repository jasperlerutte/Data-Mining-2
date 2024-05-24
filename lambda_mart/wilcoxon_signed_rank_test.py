import os
import time
import xgboost as xgb
from utils.load_data import load_train_val_test_split
from dotenv import load_dotenv
from lambda_mart.normalized_discounted_cumulative_gain import ndcg_weighted
from scipy.stats import wilcoxon
import gc

load_dotenv()

YOUR_TRAIN_FILE = os.getenv("TRAIN_FILE")
YOUR_BASIC_TRAIN_FILE = os.getenv("BASIC_TRAIN_FILE")

basic_model_path = r"C:\Users\robbe\PycharmProjects\Data-Mining-2\lambda_mart\model_checkpoints\basic_model.json"
engineered_model_path =  r"C:\Users\robbe\PycharmProjects\Data-Mining-2\lambda_mart\model_checkpoints\engineer_model.json"

# Timing the engineered set loading and predictions
print(f"Running predictions on own test set.")
start_time_engineered = time.time()

_, _, test_data = load_train_val_test_split(YOUR_TRAIN_FILE,
                                            drop_original_targets=True, seed=420,
                                            drop_extra_columns=["random_bool"])

X_test_engineered = test_data.drop(columns=["label"])

engineered_model = xgb.XGBRanker()
engineered_model.load_model(engineered_model_path)
engineered_scores = engineered_model.predict(X_test_engineered)

end_time_engineered = time.time()
elapsed_time_engineered = end_time_engineered - start_time_engineered
print(f"Elapsed time for loading and predicting engineered set: {elapsed_time_engineered} seconds")

del X_test_engineered, engineered_model, test_data
gc.collect()

# Timing the basic set loading and predictions
print(f"Loading basic set...")
start_time_basic = time.time()

_, _, test_data = load_train_val_test_split(YOUR_BASIC_TRAIN_FILE,
                                            drop_original_targets=True, seed=420)
y_test = test_data["label"]
X_test = test_data.drop(columns=["label"])

basic_model = xgb.XGBRanker()
basic_model.load_model(basic_model_path)
basic_scores = basic_model.predict(X_test)

end_time_basic = time.time()
elapsed_time_basic = end_time_basic - start_time_basic
print(f"Elapsed time for loading and predicting basic set: {elapsed_time_basic} seconds")

del basic_model
gc.collect()

X_test['basic_score'] = basic_scores
X_test['engineered_score'] = engineered_scores

# Timing the NDCG calculation and statistical test
print(f"Calculating NDCGs and performing statistical test...")
start_time_ndcg = time.time()

basic_ndcgs = []
engineered_ndcgs = []
for qid in X_test['qid'].unique():
    qid_rows = X_test[X_test['qid'] == qid]
    qid_basic_scores = qid_rows['basic_score']
    qid_engineered_scores = qid_rows['engineered_score']
    qid_labels = y_test[qid_rows.index]
    basic_ndcgs.append(ndcg_weighted(qid_basic_scores, qid_labels))
    engineered_ndcgs.append(ndcg_weighted(qid_engineered_scores, qid_labels))

stat, p = wilcoxon(basic_ndcgs, engineered_ndcgs)

end_time_ndcg = time.time()
elapsed_time_ndcg = end_time_ndcg - start_time_ndcg
print(f"Elapsed time for NDCG calculation and statistical test: {elapsed_time_ndcg} seconds")

print(f"Statistics={stat}, p={p}")
