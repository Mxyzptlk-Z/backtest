import numpy as np
import os
import logging

class Client:

    def __init__(self, init_capital = 1e6, slippage = 0.0001 , commission = 0.0003, stamp_duty = 0.001, ub = np.Inf, lb = -0.08):
        """
        This method initializes the parameters entering the Client object.
        """
        self.init_capital = init_capital
        self.slippage = slippage
        self.commission = commission  # bilateral commission
        self.stamp_duty = stamp_duty  # default 0.1%
        self.upper_bound = ub
        self.lower_bound = lb

    def __repr__(self):
        """
        This method returns a string representation of a Client object.
        """
        return 'Client_Account, init_capital=%f, slippage=%f, commission=%f' % (self.init_capital, self.slippage, self.commission)
    
    def apply_slippage(self, price, pct_slippage, slippage_type):
        """
        This method defines the rate and type of slippage during transaction.
        Available slippage type: 'byRate' and 'byVolume'
        """
        if slippage_type == 'byRate':
            price_with_slippage = price * (1 + pct_slippage)
        elif slippage_type == 'byVolume':
            price_with_slippage = price + pct_slippage
        else:
            print('Please indicate which slippage type.')
        return price_with_slippage

    def loss_lim(self, price, ret):
        """
        This method sets the boundary of loss or profit and manipulates with price.
        """
        PROFIT_LIMIT = np.array(np.where(ret>=self.upper_bound))
        LOSS_LIMIT = np.array(np.where(ret<=self.lower_bound))
        price_filter = price.copy()
        for idx in np.hstack([LOSS_LIMIT, PROFIT_LIMIT]).T:
            price_filter.iloc[idx[0]+2:,idx[1]] = price.iloc[idx[0]+1,idx[1]]
        return price_filter

    def pnl_lim(self, ret):
        """
        This method sets the boundary of loss or profit and manipulates with return.
        """
        PROFIT_LIMIT = np.array(np.where(ret>=self.upper_bound))
        LOSS_LIMIT = np.array(np.where(ret<=self.lower_bound))
        ret_filter = ret.copy()
        for idx in np.hstack([LOSS_LIMIT, PROFIT_LIMIT]).T:
            ret_filter.iloc[idx[0]+1:,idx[1]] = 0
        return ret_filter

    def log(self):
        """
        This method generates the log from client's operation.
        """
        LOG_PATH = './logs'
        if os.path.exists(LOG_PATH) is False:
            os.makedirs(LOG_PATH)
        logging.basicConfig(filename=os.path.join(LOG_PATH, self.index_code + '_account activities.log'), level=logging.INFO, format='%(message)s\n', force=True)
    