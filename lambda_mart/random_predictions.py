import numpy as np
from utils.load_data import load_train_val_test_split
from lambda_mart.normalized_discounted_cumulative_gain import ndcg_weighted


# load our own holdout set again and get predictions
_, _, random_predictions_data = load_train_val_test_split(
    "training_set_VU_DM.csv",
    drop_original_targets=True, seed=420,
    frac_test=1
)

y_test = random_predictions_data["label"]
random_predictions_data = random_predictions_data.drop(columns=["label"])
random_predictions_data['score'] = np.random.rand(random_predictions_data.shape[0])

ndcg_score = 0
n_queries = 0
for qid in random_predictions_data['qid'].unique():
    qid_rows = random_predictions_data[random_predictions_data['qid'] == qid]
    qid_scores = qid_rows['score']
    qid_labels = y_test[qid_rows.index]
    ndcg_score += ndcg_weighted(qid_scores, qid_labels)
    n_queries += 1

ndcg_score /= n_queries

print(f"NDCG score of random predictions on complete set of {n_queries} queries: ",
      ndcg_score)
