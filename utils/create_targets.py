import pandas as pd


def add_target_column(data: pd.DataFrame, booking_score: int = 5, clicking_score: int = 1) -> pd.DataFrame:
    data['label'] = data.apply(lambda row: booking_score if row['booking_bool'] else (clicking_score if row['click_bool'] else 0), axis=1)
    return data