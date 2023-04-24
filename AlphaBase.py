import numpy as np
import pandas as pd
import os
import datetime
from Util import Util

class AlphaBase(Util):

    def __init__(self, index_code, start_date, end_date, max_stocks, date_rule):
        """
        This method initializes the parameters entering the AlphaBase object.
        """
        super().__init__(index_code, start_date, end_date, max_stocks, date_rule)
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
    
    def alpha_rule(self, date, stock):
        """
        This method defines the alpha rule (calculation of alpha) for each day.
        """
        # pe = self.acct[stock].loc[date, 'pe_ttm']
        data = self.component[stock]
        wms = ((data[date-datetime.timedelta(days=self.window):date]['hfq_high'].max() - data.loc[date]['hfq_close']) / (data[date-datetime.timedelta(days=self.window):date]['hfq_high'].max() - data[date-datetime.timedelta(days=self.window):date]['hfq_low'].min())) * 100
        return -float(wms)
    
    def apply_alpha(self):
        """
        This method applies the alpha rule to the whole backtest period and generates the score.
        """
        stocks = list(self.component.keys())
        dates = self.index[self.start_date : self.end_date].index
        score = pd.DataFrame(columns = stocks, index = dates)
        for date in dates:
            for stock in stocks:
                score.loc[date, stock] = self.alpha_rule(date, stock)
        return score, self.index, self.component, self.dividend

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
        
        # pe = ((pe - pe.min()) / (pe.max() - pe.min()))[self.start_date:self.end_date]
        # wms = ((wms - wms.min()) / (wms.max() - wms.min()))[self.start_date:self.end_date]
        # roc = ((roc - roc.min()) / (roc.max() - roc.min()))[self.start_date:self.end_date]
        # bias = ((bias - bias.min()) / (bias.max() - bias.min()))[self.start_date:self.end_date]

        # pe = ((pe - pe.rolling(200).mean()) / pe.rolling(200).std())[self.start_date:self.end_date]
        # wms = ((wms - wms.rolling(200).mean()) / wms.rolling(200).std())[self.start_date:self.end_date]
        # roc = ((roc - roc.rolling(200).mean()) / roc.rolling(200).std())[self.start_date:self.end_date]
        # bias = ((bias - bias.rolling(200).mean()) / bias.rolling(200).std())[self.start_date:self.end_date]
        # score = - 0.0415 * pe + 0.1512 * wms - 0.1427 * roc - 0.6652 * bias
        score = pe - (wms + roc + bias)
    
        return score, self.index, self.component, self.dividend
