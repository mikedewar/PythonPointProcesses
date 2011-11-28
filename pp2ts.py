import numpy as np
import datetime as dt
import pandas
from scipy.stats import kde
from scipy.special import gamma


class Method(object):
    
    # base class for a rate estimation method
    
    def __init__(self, pp):
        self.pp = np.array(pp, dtype=float)
        self.pp.sort()
        
    def sample(self):
        raise NotImplementedError

class Binning(Method):
    
    def __init__(self, pp):
        """
        The standard binning method (don't use this)
        
        pp : list
            the timestamps of events
        """
        Method.__init__(self, pp)
    
    def sample(self, sample_times):
        # we have to do some futzing to make sure the bin edges are either
        # side of the sample times. We make the assumption that the sample
        # times are regular.
        half_period = (sample_times[1] - sample_times[0])/2
        # shift all sample times back by half the sampling period
        edges = [t-half_period for t in sample_times]
        # and then add the final edge on the end
        edges.append(sample_times[-1]+half_period)
        counts, edges = np.histogram(self.pp, edges)
        return counts
        
class Kernel(Method):
    
    def __init__(self, pp):
        """
        A Kernel Density Estimate based method.
        
        pp : list
            the timestamps of events
        """
        if len(pp) < 10:
            raise ValueError('not enough clicks to form density')
        Method.__init__(self, pp)
        self.model = kde.gaussian_kde(self.pp)

    def sample(self, sample_times):
        y = self.model.evaluate(sample_times)
        # we scale up so that the sum of the series is the same as the number
        # of clicks
        y = y * (len(self.pp)/sum(y))
        return y

class MovingAverage(Method):
    
    def __init__(self,pp,w=5):
        """
        A moving average window.
        
        pp : list
            the timestamps of events
        
        w : int
            width of the moving average window
        """
        self.w = w
        Method.__init__(self, pp)
    
    def sample(self, sample_times):
        
        def window(X):
            y = []
            for x in X:
                if abs(x) > self.w:
                    y.append(0)
                else:
                    y.append(1./self.w)
            return sum(y)
        
        out = np.array([
            window(t - self.pp[self.pp < t]) 
            for t in sample_times
        ], dtype=float)
        out = out * (len(self.pp)/sum(out)) 
        return out

class GammaWindow(Method):
    
    def __init__(self, pp, theta=2.0, k=2.0):
        Method.__init__(self, pp)
        self.theta = theta
        self.k = k
    
    def sample(self, sample_times):
        den = gamma(self.k) * (self.theta**self.k)
        clicks = self.pp
        times = np.array(sample_times)
        clicks.shape = (len(clicks), 1)
        delta = times - clicks
        delta[delta < 0] = 0
        return sum((delta)**(self.k-1)*np.exp(-(delta)/self.theta) / den, 0)
        


def pp2ts(pp, method=Kernel, period=1, t0=None, N=None, **kwargs):
    """
    convert a point process into a time series
    
    pp : list
        point process as a list of timestamps
        
    period: float
        time in seconds between samples
    
    method : Method object
        conversion method
    
    t0 : timestamp
        initial time
    
    N : int
        number of samples to generate (optional)
    
    Returns
    
    ts : a pandas.TimeSeries object
        time series
    """
    
    if t0 is None:
        t0 = min(pp)
        
    if N is None:
        N = int(round((max(pp) - min(pp)) / float(period)))
    
    # bulid model
    model = method(pp, **kwargs)
    # create sampletimes
    sample_times = range(t0, period*N, period)
    # sample
    samples = model.sample(sample_times)
    # form Time Series
    out = pandas.TimeSeries(samples, sample_times)
    # convert timestamp indices to datetime objects
    out = out.rename(dt.datetime.fromtimestamp)
    return out


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    clicks = np.random.randint(0,3*60*60,300)
    T = 600
    binned_series = pp2ts(clicks, Binning, period = T)
    kde_series = pp2ts(clicks, Kernel, period = T)
    ma_series_2m = pp2ts(clicks, MovingAverage, period = T, w=2*60)
    ma_series_1h = pp2ts(clicks, MovingAverage, period = T, w=60*60)
    gamma_series = pp2ts(clicks, GammaWindow, period=T)
    
    binned_series.plot(label="bins")
    kde_series.plot(label="kde")
    ma_series_2m.plot(label="2m moving average")
    ma_series_1h.plot(label="1h moving average")
    gamma_series.plot(label="gamma window")
    
    plt.plot([dt.datetime.fromtimestamp(c) for c in clicks], [0 for i in clicks], 'kx')


    plt.legend()
    
    plt.show()
    
    
    
    
    
    
    
