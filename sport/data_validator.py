# open source pakcages
import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> None:
    if df is None:
        raise ValueError("Expected data frame is None")
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"""Expected data frame is not of type pandas.DataFrame: {type(df)}""")
    if df.shape[0] == 0:
        raise ValueError("Expected data frame is empty")

    for col in df.columns:
        if df[col].isna().any():
            raise ValueError(f"""Column {col} of expected data frame has NA values""")

