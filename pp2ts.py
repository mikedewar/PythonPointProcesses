import numpy as np
import datetime as dt
import pandas
from scipy.stats import kde
        
class Kernel(object):
    
    def __init__(self, pp):
        """
        A Kernel Density Estimate based method.
        
        pp : list
            the timestamps of events
        """
        if len(pp) < 10:
            raise ValueError('not enough clicks to form density')
        self.pp = np.array(pp, dtype=float)
        self.pp.sort()
        self.model = kde.gaussian_kde(self.pp)

    def sample(self, sample_times):
        y = self.model.evaluate(sample_times)
        # we scale up so that the sum of the series is the same as the number
        # of clicks
        y = y * (len(self.pp)/sum(y))
        return y


def pp2ts(pp, period=1, t0=None, N=None, **kwargs):
    """
    estimate the rate of a point process, returning the result as a pandas TimeSeries
    
    pp : list
        point process as a list of unix timestamps
        
    period: float
        time in seconds between output samples
    
    t0 : timestamp
        initial time (optional) to generate output samples
    
    N : int
        number of output samples to generate (optional)
    
    Returns
    
    ts : a pandas.TimeSeries object
        time series
    """
    
    if t0 is None:
        t0 = min(pp)
        
    if N is None:
        N = int(round((max(pp) - min(pp)) / float(period)))
    
    # bulid model
    model = Kernel(pp, **kwargs)
    # create sampletimes
    sample_times = range(t0, t0+period*N, period)
    # sample
    samples = model.sample(sample_times)
    # form Time Series
    out = pandas.TimeSeries(samples, sample_times)
    # convert timestamp indices to datetime objects
    out = out.rename(dt.datetime.fromtimestamp)
    return out
