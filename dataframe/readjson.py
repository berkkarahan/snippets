import json
import pandas as pd
from pandas.io.json import json_normalize

def load_frame(csv="", json_cols=[], nrows=None):
    df = pd.read_csv(
    csv,
    converters={column: json.loads for column in json_cols},
    nrows=nrows
    )

    for column in json_cols:
        col_as_df = json_normalize(df[column])
        col_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in col_as_df.columns]
        df = df.drop(column, axis=1).merge(col_as_df, right_index=True, left_index=True)
        print(f"Loaded {os.path.basename(csv)}. Shape: {df.shape}")
    return df
