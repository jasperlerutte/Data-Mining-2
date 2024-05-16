import pandas as pd
from utils.load_data import load_data


def add_difference_features(data: pd.DataFrame, feature_names: list[str], target_name: str) -> pd.DataFrame:
    for feature_name in feature_names:
        # find mean of all values of the feature where target_name == 1
        mean_feature = data.loc[data[target_name] == 1, feature_name].mean()
        data[f"diff_{feature_name}_{target_name}"] = data[feature_name] - mean_feature

    return data

if __name__ == "__main__":
    differences_features = ['prop_starrating', 'prop_review_score', 'prop_location_score1']
    data = load_data("training_set_VU_DM.csv", n_rows=10000)
    data = add_difference_features(data, differences_features, 'booking_bool')
    print(data.head())






