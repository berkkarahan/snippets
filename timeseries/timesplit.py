#Expecting X and y as pandas objects(DataFrame or Series)
def ts_train_test_split(X, y=None, test_size=0.25):

    test_idx = int(len(X)*(1-test_size))

    X_train = X.iloc[:test_idx]
    X_test = X.iloc[test_idx:]

    if y is None:
        return X_train, X_test
    else:
        y_train = y.iloc[:test_idx]
        y_test = y.iloc[test_idx:]
        return X_train, X_test, y_train, y_test
