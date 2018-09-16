

def filter_constants(df):
    const = [c for c in df.columns if df[c].nunique()==1]
    return df.drop(const, axis=1)
