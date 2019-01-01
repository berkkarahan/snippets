import numpy as np
import threading
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
import gc

#X and Y should be numpy arrays
def fit_model_cv(mdl, x, y, error_calc=mean_squared_error, nsplits=5, cv='kfold'):
    def _fitmodel(mdl,x,y):
        return mdl.fit(x,y)

    if cv == 'kfold':
        cvgen = KFold(n_splits=nsplits)
    elif cv == 'time':
        cvgen = TimeSeriesSplit(n_splits=nsplits)
    else:
        raise TypeError("cv arguement should either take 'kfold' or 'time'.")
    threadlist = []
    modlist = []
    train_ind = []
    holdout_ind = []
    predictions = []
    score = []

    for tr_i, ho_i in cvgen.split(x, y):
        train_ind.append(tr_i)
        holdout_ind.append(ho_i)
        cloned_mdl = clone(mdl)
        modlist.append(cloned_mdl)
        task = threading.Thread(target=_fitmodel, args=(cloned_mdl,x[tr_i],y[tr_i],))
        threadlist.append(task)

    for t in threadlist:
        t.start()

    for t in threadlist:
        t.join()

    for i in range(0,nsplits):
        m = modlist[i]
        predictions.append(m.predict(x[holdout_ind[i]]))
        score.append(error_calc(y[holdout_ind[i]], predictions[i]))
    del threadlist, train_ind, holdout_ind, predictions
    gc.collect()
    print("Average error for models are: {}".format(np.mean(score)))
    return modlist


def predict_from_models(model_list, X):
    def _predict(model_list, X):
        infoldpreds = []
        for m in model_list:
            infoldpreds.append(m.predict(X))
        infoldpreds = np.hstack(infoldpreds)
        return infoldpreds
    infoldpreds = _predict(model_list, X)
    return np.mean(infoldpreds, axis=1)
