import numpy as np
import threading
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
import gc

#X and Y should be numpy arrays
def fit_model_cv(mdl, x, y, cv=5):
    def _fitmodel(mdl,x,y):
        return mdl.fit(x,y)

    kfold = KFold(n_splits=cv)
    threadlist = []
    modlist = []
    train_ind = []
    holdout_ind= []
    predictions = []
    score = []

    for tr_i, ho_i in kfold.split(x,y):
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

    for i in range(0,5):
        m = modlist[i]
        predictions.append(m.predict(x[holdout_ind[i]]))
        score.append(mean_squared_error(y[holdout_ind[i]], predictions[i]))
    del threadlist, train_ind, holdout_ind, predictions
    gc.collect()
    print("Average mean_sq_error for models are: {}".format(np.mean(score)))
    return modlist
