import numpy as np
import scipy.optimize as sco
import statsmodels.api as sm
from Util import Util

# Portfolio Optimization module

class PortOpt(Util):

    def __init__(self, index_code, start_date, end_date, max_stocks):
        """
        This method initializes the parameters entering the PortOpt object.
        """
        Util.__init__(self, index_code, start_date, end_date)
        self.max_stocks = max_stocks
    
    def __repr__(self):
        """
        This method returns a string representation of a Port_Opt object.
        """
        return 'portfolio_optimization'

    def gen_single_index(self, stock, market):
        """
        This method generates the single index estimator.
        """
        y = stock.dropna()
        X = market.dropna()
        X = sm.add_constant(X)
        est = sm.OLS(y,X).fit()
        beta = est.params.loc['close']
        beta = np.array(beta).reshape(-1,1)
        resid_var = (y - np.array(est.predict(X))).var()
        
        mkt_var = market.var()
        sgl_idx = mkt_var * np.dot(beta, beta.T) + np.diag(resid_var)
        return sgl_idx
    
    def covariance_estimator(self, in_sample, mkt_in_sample):
        """
        This method generates the covariance matrix estimation.
        """
        cov = 1/3 * in_sample.cov() + 1/3 * np.diag(in_sample.var().tolist()) + 1/3 * self.gen_single_index(in_sample, mkt_in_sample)
        return cov

    def min_func_sharpe(self, weights, in_sample, mkt_in_sample):
        """
        This method defines the target function.
        """
        return - np.sum(in_sample.mean() * weights) * 252 + 20 * np.sqrt(np.dot(weights.T, np.dot(self.covariance_estimator(in_sample, mkt_in_sample) * 252, weights)))
    
    def res(self, in_sample, mkt_in_sample):
        """
        This method returns the result of portfolio optimization, i.e., portfolio weights.
        """
        cons = ({'type':'eq','fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for x in range(self.max_stocks))
        eweights = np.array(self.max_stocks * [1./self.max_stocks])
        opts = sco.minimize(self.min_func_sharpe, eweights, method='SLSQP', args = (in_sample, mkt_in_sample), bounds=bnds, constraints=cons)
        w = opts['x']
        return w
            