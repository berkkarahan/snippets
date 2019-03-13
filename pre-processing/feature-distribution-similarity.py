import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def remove_duplicate_columns(df):
    colsToRemove = []
    colsScaned = []
    dupList = {}
    columns = df.columns
    for i in range(len(columns)-1):
        v = train_df[columns[i]].values
        dupCols = []
        for j in range(i+1,len(columns)):
            if np.array_equal(v, train_df[columns[j]].values):
                colsToRemove.append(columns[j])
                if columns[j] not in colsScaned:
                    dupCols.append(columns[j])
                    colsScaned.append(columns[j])
                    dupList[columns[i]] = dupCols
    colsToRemove = list(set(colsToRemove))
    df.drop(colsToRemove, axis=1, inplace=True)
    print(f">> Dropped {len(colsToRemove)} duplicate columns")

def different_columns(train_df, test_df, threshold=0.1):
    """Use KS to estimate columns where distributions differ a lot from each other"""

    # Find the columns where the distributions are very different
    diff_data = []
    for col in train_df.columns:
        statistic, pvalue = ks_2samp(
            train_df[col].values,
            test_df[col].values
        )
        if pvalue <= 0.05 and np.abs(statistic) > threshold:
            diff_data.append({'feature': col, 'p': np.round(pvalue, 5), 'statistic': np.round(np.abs(statistic), 2)})

    # Put the differences into a dataframe
    diff_df = pd.DataFrame(diff_data).sort_values(by='statistic', ascending=False)

    return diff_df

def get_similar_features(train_df, test_df, threshold=0.1):
    diff_df = different_columns(train_df, test_df, threshold)
    train_new = train_df.copy().drop(diff_df.feature.values, axis=1)
    test_new = test.df.copy().drop(diff_df.feature.values, axis=1)
    return train_new, test_new
