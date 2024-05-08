# importing os module for environment variables
import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv, dotenv_values

import sys
print(sys.path)


# loading variables from .env file
load_dotenv()
path = os.getenv("DATA_FOLDER")

df = pd.read_csv(path + 'training_set_VU_DM.csv')
# get dimensions of the dataframe
print(df.shape)
print(df.columns)

# drop the columns with more than 10000 missing values
df = df.dropna(thresh=10000, axis=1)
df = df.drop('date_time', axis=1)
print(df.corr())

plt.matshow(df.corr())
plt.xticks(range(len(df.columns)), labels=df.columns, rotation=90)
plt.yticks(range(len(df.columns)), labels=df.columns)
plt.show()
plt.clf()
