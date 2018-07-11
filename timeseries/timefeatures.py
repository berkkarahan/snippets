import pandas as pd

class IndexTypeError(Exception):
    pass

class ObjectTypeError(Exception):
    pass

class TimeTransformer(object):
    def __init__(self, timeseries):
        
        self._timeseries = timeseries
        
        
    def _check(self):
        
        if not(isinstance(self.timeseries, pd.DataFrame)):
            raise ObjectTypeError("Given dataset is not of type pandas.DataFrame")
        
        if not(isinstance(self.timeseries.index, pd.DatetimeIndex)):
            raise IndexTypeError("Index of given dataset is not of type pandas.DateTimeIndex")
        
    def _timefeatures(self):

        self._timeseries["Year"] = self._timeseries.Date.apply(lambda x: x.year)
        self._timeseries["Month"] = self._timeseries.Date.apply(lambda x: x.month)
        self._timeseries["Hour"] = self._timeseries.Date.apply(lambda x: x.hour)
        self._timeseries["WeekDay"] = self._timeseries.Date.apply(lambda x: x.weekday())
        self._timeseries["DayCount"] = self._timeseries.Date.apply(lambda x: x.toordinal())

        self._timeseries = pd.get_dummies(self._timeseries, columns=["Month","Hour","WeekDay"])

        return self
    
    def transform(self):
        
        self._check()
        self._timefeatures()
        
        return self._timefeatures
