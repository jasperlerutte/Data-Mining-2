# importing os module for environment variables
import os
# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values


# loading variables from .env file
load_dotenv()

# accessing and printing value
path = os.getenv("DATA_PATH")
print(os.getenv("DATA_PATH"))