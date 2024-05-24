import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get the current working directory of the script and read data
current_dir = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
# csv_train = os.path.join(current_dir, 'Data', 'training_set_VU_DM.csv')
# csv_test = os.path.join(current_dir, 'Data', 'test_set_VU_DM.csv')

# # Read the CSV file into a DataFrame
# train = pd.read_csv(csv_train)
# test = pd.read_csv(csv_test)

# # combine train and test data and add column to indicate whether row is from train or test
# train['is_train'] = 1
# test['is_train'] = 0
# df = pd.concat([train, test], axis=0)
df = pd.read_csv('C:/Users/esrio_0v2bwuf/Desktop/Master_AI/Data_Mining_Techniques/Assignments/Assignment2/Data-Mining-2/feature_engineering/Data/no_missing_values.csv')

df['month'] = pd.to_datetime(df['date_time']).dt.month
print(df.head())

# print the column names of df
# print(train.columns)
# print(test.columns)
# print(df.columns)


def normalise(data, cols, respect):
    for col in cols:
        # data[f'{col}_norm_{respect}'] = data.groupby(respect)[col].transform(lambda x: (x - x.mean()) / x.std())
        data[f'{col}_norm_{respect}'] = data.groupby(respect)[col].transform(lambda x: 0 if x.std() == 0 else (x - x.mean()) / x.std())
    return data


def log_transform(data, cols):
    for col in cols:
        data[col] = np.log1p(data[col])
    return data


# normalise 'prop_log_historical_price' and 'price_usd' columns with respect to 'srch_id'
log_transform_cols = ['price_usd', 'orig_destination_distance', 'comp1_rate_percent_diff','comp2_rate_percent_diff',
                      'comp3_rate_percent_diff', 'comp4_rate_percent_diff', 'comp5_rate_percent_diff',
                      'comp6_rate_percent_diff', 'comp7_rate_percent_diff', 'comp8_rate_percent_diff']

norm_cols = ['prop_log_historical_price', 'price_usd', 'orig_destination_distance', 'comp1_rate_percent_diff',
             'comp2_rate_percent_diff', 'comp3_rate_percent_diff', 'comp4_rate_percent_diff', 'comp5_rate_percent_diff',
             'comp6_rate_percent_diff', 'comp7_rate_percent_diff', 'comp8_rate_percent_diff']

wrt_cols = ['srch_id', 'prop_id', 'month', 'srch_booking_window', 'srch_destination_id', 'prop_country_id']

df_norm = df.copy()
df_norm = log_transform(df_norm, log_transform_cols)

print('finished with log transformations')

for wrt in wrt_cols:
    df_norm = normalise(df_norm, norm_cols, wrt)

print('finished with normalizations')

# df.to_csv(os.path.join(current_dir, 'Data', 'df.csv'), index=False)
# print('finished with df.csv')

df_norm.to_csv(os.path.join(current_dir, 'Data', 'df_norm.csv'), index=False)
print('finished with df_norm.csv')