import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv


def plot_booking_percentage_feature(df: pd.DataFrame, feature_name: str, n_bins: int = 10):
    # create bins
    bins = pd.cut(df[feature_name], bins=n_bins, include_lowest=True)

    # calculate booking percentage per bin
    booking_percentage = df.groupby(bins)["booking_bool"].mean()

    # plot
    booking_percentage.plot(kind="bar")
    plt.xlabel(feature_name)
    plt.ylabel("Booking percentage")
    plt.title(f"Booking percentage per {feature_name}")
    plt.show()


if __name__ == "__main__":
    load_dotenv()
    fil_path = os.getenv("DATA_FOLDER")
    data_file = "training_set_VU_DM.csv"
    data_path = os.path.join(fil_path, data_file)

    df = pd.read_csv(data_path)

    # Adding a column for the logarithm of price_usd
    df['log_price_usd'] = df['price_usd'].apply(lambda x: np.log(x) if x > 0 else 0)

    # Normalizing price_usd with respect to other observations with the same src_id
    df['normalized_log_price_usd'] = df.groupby("srch_id")["log_price_usd"].transform(
        lambda x: (x - x.mean()) / x.std())

    plot_booking_percentage_feature(df, 'normalized_log_price_usd')
