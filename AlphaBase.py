import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from Util import Util

class AlphaBase(Util):

    def __init__(self, index_code, start_date, end_date):
        """
        This method initializes the parameters entering the AlphaBase object.
        """
        Util.__init__(self,index_code, start_date, end_date)
        if os.path.exists(self.IDX_PATH):
            index = self.load_data('index.pkl')
            component = self.load_data('component.pkl')
            acct = self.load_data('acct.pkl')
            self.dividend = self.load_data('dividend.pkl')
            self.index, self.component, self.acct = self.cache_update(index, component, acct)
        else:
            self.index, self.component, self.acct, self.dividend = self.fetch_data()
        
        self.window = 12
        self.timestamp = 30
    
    def __repr__(self):
        """
        This method returns a string representation of an AlphaBase object.
        """
        return 'alpha_research'

    def factor_mining(self):
        """
        This method provides with another approach generating the stock scores by batch.
        """
        pe = pd.DataFrame({tic: data['pe_ttm'] for tic, data in self.acct.items()})
        wms = pd.DataFrame({tic: ((data['hfq_high'].rolling(self.window).max() - data['hfq_close']) / (data['hfq_high'].rolling(self.window).max() - data['hfq_low'].rolling(self.window).min())) * 100 for tic, data in self.component.items()})
        roc = pd.DataFrame({tic: (data['hfq_close'] - data['hfq_close'].shift(self.window)) / data['hfq_close'].shift(self.window) for tic, data in self.component.items()})
        bias = pd.DataFrame({tic: ((data['hfq_close'] - data['hfq_close'].rolling(self.window).mean()) / data['hfq_close'].rolling(self.window).mean()) * 100 for tic, data in self.component.items()})
        
        pe = ((pe - pe.rolling(self.timestamp).min()) / (pe.rolling(self.timestamp).max() - pe.rolling(self.timestamp).min()))[self.start_date:self.end_date]
        wms = ((wms - wms.rolling(self.timestamp).min()) / (wms.rolling(self.timestamp).max() - wms.rolling(self.timestamp).min()))[self.start_date:self.end_date]
        roc = ((roc - roc.rolling(self.timestamp).min()) / (roc.rolling(self.timestamp).max() - roc.rolling(self.timestamp).min()))[self.start_date:self.end_date]
        bias = ((bias - bias.rolling(self.timestamp).min()) / (bias.rolling(self.timestamp).max() - bias.rolling(self.timestamp).min()))[self.start_date:self.end_date]

        # n = self.window
        # ncskew = pd.DataFrame({tic: (- (n * (n-1)) ** 1.5 * data['20211201':self.end_date]['hfq_close'].rolling(n).apply(lambda x: ((x - x.mean()) ** 3).sum())) / ((n-1) * (n-2) * data['20211201':self.end_date]['hfq_close'].rolling(n).apply(lambda x: ((x - x.mean()) ** 2).sum() ** 1.5)) for tic, data in tqdm(self.component.items())})
        # duvol = pd.DataFrame({tic: np.log((data['hfq_close'].rolling(n).apply(lambda x: len(x[x>x.mean()])) - 1) * data['hfq_close'].rolling(n).apply(lambda x: ((x[x<x.mean()] - x.mean()) ** 2).sum()) / (data['hfq_close'].rolling(n).apply(lambda x: len(x[x<x.mean()])) - 1) * data['hfq_close'].rolling(n).apply(lambda x: ((x[x>x.mean()] - x.mean()) ** 2).sum())) for tic, data in tqdm(self.component.items())})
        # ncskew = ((ncskew - ncskew.rolling(self.timestamp).min()) / (ncskew.rolling(self.timestamp).max() - ncskew.rolling(self.timestamp).min()))[self.start_date:self.end_date]
        # duvol = ((duvol - duvol.rolling(self.timestamp).min()) / (duvol.rolling(self.timestamp).max() - duvol.rolling(self.timestamp).min()))[self.start_date:self.end_date]
        
        score = 0.0415 * pe - 0.1512 * wms + 0.1427 * roc + 0.6652 * bias
    
        return score, self.index, self.component, self.dividend
