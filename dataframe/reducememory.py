import numpy as np


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of properties dataframe is :",start_mem," MB")
    for col in df.columns:
        print("******************************")
        print("Column: ", col)
        print("dtype before: ", df[col].dtype)

        IsInt = False
        mx = df[col].max()
        mn = df[col].min()

        # test if column can be converted to an integer
        asint = df[col].fillna(0).astype(np.int64)
        result = (df[col] - asint)
        result = result.sum()
        if result > -0.01 and result < 0.01:
            IsInt = True

        # Make Integer/unsigned Integer datatypes
        if IsInt:
            if mn >= 0:
                if mx < 255:
                    df[col] = df[col].astype(np.uint8)
                elif mx < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif mx < 4294967295:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

        # Make float datatypes 32 bit
        else:
            df[col] = df[col].astype(np.float32)

        # Print new column type
        print("dtype after: ", df[col].dtype)
        print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2
    print("Memory usage is: ", mem_usg," MB")
    print(1 - 100*mem_usg/start_mem, "% memory freed.")
    return df
