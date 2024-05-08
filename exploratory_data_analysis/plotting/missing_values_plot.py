import os
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

# accessing and printing value
folder = os.getenv("DATA_FOLDER")
data_file = "training_set_VU_DM.csv"
data_path = os.path.join(folder, data_file)

print(data_path)

# read the first 200 rows of the dataset
df = pd.read_csv(data_path, nrows=10000)
print(df.head())

# make a barchart of missing values per column
missing_values = df.isnull().sum()

# turn into percentage and plot
missing_values_percentage = missing_values / len(df)
missing_values_percentage = missing_values_percentage.sort_values(ascending=False)
plt.subplots_adjust(bottom=0.5)
missing_values_percentage.plot(kind='bar', figsize=(10, 5))
plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0%', '25%', '50%', '75%', '100%'])
plt.xticks(rotation=75, ha='right', fontsize=8)
plt.show()

