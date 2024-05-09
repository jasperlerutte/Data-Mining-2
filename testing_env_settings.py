# importing os module for environment variables
import os
import pandas as pd
# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values


# loading variables from .env file
load_dotenv()

# accessing and printing value
folder = os.getenv("DATA_FOLDER")
data_file = "training_set_VU_DM.csv"
data_path = os.path.join(folder, data_file)

print(data_path)

df = pd.read_csv(data_path)
print(df.head())