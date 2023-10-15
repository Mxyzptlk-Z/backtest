import pandas as pd
import numpy as np
import tushare as ts
import datetime
import holidays
import chinese_calendar as calendar
from dateutil.relativedelta import relativedelta
import os
import pickle
from tqdm import tqdm
from Client import Client

class Util(Client):
    
    def __init__(self, index_code, start_date, end_date):
        """
        This method initializes the parameters entering the Util object.
        """
        super().__init__()
        self.index_code = index_code
        self.start_date = start_date
        self.end_date = end_date
        self.PATH = './data cache'
        self.IDX_PATH = os.path.join(self.PATH, index_code.replace('.', '_'))

        token = 'c336245e66e2882632285493a7d0ebc23a2fbb7392b74e4b3855a222'
        ts.set_token(token)
        self.pro = ts.pro_api()
    
    def __repr__(self):
        """
        This method returns a string representation of a Util object.
        """
        return 'utility_functions'

    def save_data(self, filename, *args):
        """
        This method stores data cache to the current directory.
        """
        if not os.path.exists(self.IDX_PATH):
            os.makedirs(self.IDX_PATH)
        with open (os.path.join(self.IDX_PATH, filename), 'wb') as f:
            pickle.dump(args, f)
    
    def load_data(self, filename):
        """
        This method loads data cache from the current directory.
        """
        with open(os.path.join(self.IDX_PATH, filename), 'rb') as f:
            data = pickle.load(f)    # loading pickle requires the same pandas version as saving
        return data[0]
    
    def fetch_data(self):
        """
        This method fetches market data from tushare package.
        """
        
        START = '20050101'
        TODAY = datetime.datetime.today().strftime('%Y%m%d')
        
        if os.path.exists(os.path.join(self.IDX_PATH, 'index.pkl')):
            index = self.load_data('index.pkl')
        else:
            # Index data
            query = ts.pro_bar(ts_code = self.index_code, start_date = START, end_date = TODAY, asset = 'I')
            # query = pro.index_daily(ts_code = self.index_code, start_date = START, end_date = TODAY)
            index = query[::-1]
            index.set_index('trade_date', inplace=True)
            index.index = pd.to_datetime(index.index)
            self.save_data('index.pkl', index)
        
        pool = np.unique(self.pro.index_weight(index_code = self.index_code, end_date = TODAY)['con_code'])
        
        if os.path.exists(os.path.join(self.IDX_PATH, 'component.pkl')):
            component = self.load_data('component.pkl')
        else:
            # Constituent stock data
            component = {}
            print('Fetching volume-price data:')
            for ticker in tqdm(pool, position=0):
                real = ts.pro_bar(ts_code = ticker, start_date = START, end_date = TODAY)
                qfq = ts.pro_bar(ts_code = ticker, adj='qfq', start_date = START, end_date = TODAY)
                hfq = ts.pro_bar(ts_code = ticker, adj='hfq', start_date = START, end_date = TODAY)
                qfq.drop(['ts_code', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'], axis=1, inplace=True)
                qfq.columns = ',qfq_'.join(qfq.columns).split(',')
                hfq.drop(['ts_code', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'], axis=1, inplace=True)
                hfq.columns = ',hfq_'.join(hfq.columns).split(',')
                query = pd.merge(pd.merge(real, qfq, how='outer', on='trade_date'), hfq, how='outer', on='trade_date')
                # div = pro.dividend(ts_code = ticker, end_date = TODAY, fields = 'cash_div, stk_div, ex_date')
                # div = div[div['ex_date'].notna()]
                # combine = pd.merge(query, div, how = 'outer', left_on = 'trade_date', right_on = 'ex_date')
                # query = pro.daily(ts_code = ticker, start_date = START, end_date = TODAY)
                query.index = pd.to_datetime(query['trade_date'])
                query.drop(['ts_code', 'trade_date'], axis=1, inplace=True)
                component[ticker] = query[::-1]
            self.save_data('component.pkl', component)
        
        if os.path.exists(os.path.join(self.IDX_PATH, 'acct.pkl')):
            acct = self.load_data('acct.pkl')
        else:
            # Accounting data
            acct = {}
            print('Fetching accounting data:')
            for ticker in tqdm(pool, position=0):
                query = self.pro.daily_basic(ts_code = ticker, start_date = START, end_date = TODAY, fields='ts_code, trade_date, turnover_rate, volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share, total_mv, circ_mv')
                query.index = pd.to_datetime(query['trade_date'])
                query.drop(['ts_code', 'trade_date'], axis=1, inplace=True)
                acct[ticker] = query[::-1]
            self.save_data('acct.pkl', acct)
        
        if os.path.exists(os.path.join(self.IDX_PATH, 'dividend.pkl')):
            dividend = self.load_data('dividend.pkl')
        else:
            # dividend data
            dividend = {}
            print('Fetching dividend data:')
            for ticker in tqdm(pool, position=0):
                query = self.pro.dividend(ts_code = ticker, fields = 'cash_div, stk_div, stk_bo_rate, stk_co_rate, ex_date')
                query = query[query['ex_date'].notna()]
                query['ex_date'] = pd.to_datetime(query['ex_date'])
                dividend[ticker] = query[::-1]
            # normally this takes less than 1 min to fetch, but the api only allows 500 lookup / min
            # solution: import time, time.sleep(0.1)
            self.save_data('dividend.pkl', dividend)
        
        return index, component, acct, dividend
    
    def cache_update(self, index, component, acct):
        """
        This method updates the data cache every business day.
        """
        
        TODAY = datetime.datetime.today()
        last_date = index.index[-1]
        
        if datetime.datetime.strptime(TODAY.strftime('%Y%m%d'), '%Y%m%d') > last_date and int(TODAY.strftime('%H')) >= 16 and TODAY.weekday() < 5:
            # update index data
            index_start = (last_date + datetime.timedelta(days=1)).strftime('%Y%m%d')
            query = self.pro.index_daily(ts_code = self.index_code, start_date = index_start, end_date = TODAY.strftime('%Y%m%d'))
            query.set_index('trade_date', inplace=True)
            query.index = pd.to_datetime(query.index)
            index = pd.concat([index, query[::-1]], axis=0)
        
            # update constituent stock data
            print('Updating volume-price data:')
            for ticker, comp in tqdm(component.items(), position=0):
                comp_start = (comp.index[-1] + datetime.timedelta(days=1)).strftime('%Y%m%d')
                real = ts.pro_bar(ts_code = ticker, start_date = comp_start, end_date = TODAY.strftime('%Y%m%d'))
                qfq = ts.pro_bar(ts_code = ticker, adj='qfq', start_date = comp_start, end_date = TODAY.strftime('%Y%m%d'))
                hfq = ts.pro_bar(ts_code = ticker, adj='hfq', start_date = comp_start, end_date = TODAY.strftime('%Y%m%d'))
                qfq.drop(['ts_code', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'], axis=1, inplace=True)
                qfq.columns = ',qfq_'.join(qfq.columns).split(',')
                hfq.drop(['ts_code', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'], axis=1, inplace=True)
                hfq.columns = ',hfq_'.join(hfq.columns).split(',')
                query = pd.merge(pd.merge(real, qfq, how='outer', on='trade_date'), hfq, how='outer', on='trade_date')
                query.index = pd.to_datetime(query['trade_date'])
                query.drop(['ts_code', 'trade_date'], axis=1, inplace=True)
                # component.update({ticker: comp.append(query[::-1])})
                component.update({ticker: pd.concat([comp, query[::-1]])})
            
            # update accounting data
            print('Updating accounting data:')
            for ticker, acctg in tqdm(acct.items(), position=0):
                acctg_start = (acctg.index[-1] + datetime.timedelta(days=1)).strftime('%Y%m%d')
                query = self.pro.daily_basic(ts_code = ticker, start_date = acctg_start, end_date = TODAY.strftime('%Y%m%d'), fields='ts_code, trade_date, turnover_rate, volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share, total_mv, circ_mv')
                query.index = pd.to_datetime(query['trade_date'])
                query.drop(['ts_code', 'trade_date'], axis=1, inplace=True)
                # acct.update({ticker: acctg.append(query[::-1])})
                acct.update({ticker: pd.concat([acctg, query[::-1]])})
            
            # update dividend data
            # print('Updating dividend data:')
            # for ticker, _ in tqdm(dividend.items(), position=0):
            #     query = pro.dividend(ts_code = ticker, fields = 'cash_div, stk_div, stk_bo_rate, stk_co_rate, ex_date')
            #     query = query[query['ex_date'].notna()]
            #     query['ex_date'] = pd.to_datetime(query['ex_date'])
            #     dividend.update({ticker: query[::-1]})

            # store data to cache
            self.save_data('index.pkl', index)
            self.save_data('component.pkl', component)
            self.save_data('acct.pkl', acct)
            # self.save_data('dividend.pkl', dividend)
        
        else:
            pass
            
        return index, component, acct
    
    def period_delta(self, freq):
        ONE_DAY = datetime.timedelta(days=1)
        ONE_WEEK = relativedelta(weeks=1)
        TWO_WEEK = relativedelta(weeks=2)
        ONE_MONTH = relativedelta(months=1)
        if freq == 'D':
            timedelta = ONE_DAY
        elif freq == 'W':
            timedelta = ONE_WEEK
        elif freq == '2W':
            timedelta = TWO_WEEK
        elif freq == 'M':
            timedelta = ONE_MONTH
        return timedelta

    def next_trade_day(self, date):
        ONE_DAY = datetime.timedelta(days=1)
        next_trade_day = date + ONE_DAY
        while next_trade_day.weekday() in holidays.WEEKEND or calendar.is_holiday(next_trade_day):
            next_trade_day += ONE_DAY
        return next_trade_day

    def next_trade_dates(self, dates):
        return pd.DatetimeIndex([self.next_trade_day(date) for date in dates])
    
    def exist_holiday_except_weekend(self, start_date, end_date):
        ONE_DAY = datetime.timedelta(days=1)
        next_day = start_date + ONE_DAY
        while next_day <= end_date:
            if next_day.weekday() not in holidays.WEEKEND and calendar.is_holiday(next_day):
                return True
            else:
                next_day += ONE_DAY
        return False

    def annual_rate(self, ret):
        return (1 + ret).prod() ** (252/(len(ret)+1)) - 1
    
    def Beta(self, rets):
        return rets.cov().iloc[0,1] / rets.cov().iloc[1,1]
    
    def Alpha(self, rets):
        ar_s = self.annual_rate(rets.Portfolio)
        ar_m = self.annual_rate(rets.Benchmark)
        beta = self.Beta(rets)
        return ar_s - beta * ar_m
    
    def Max_Drawdown(self, nv):
        end_idx = np.argmax((np.maximum.accumulate(nv) - nv) / np.maximum.accumulate(nv))  # end point
        if end_idx == 0:
            return 0
        start_idx = np.argmax(nv[:end_idx])    # start point
        return (nv[start_idx] - nv[end_idx]) / nv[start_idx]
    
    def Sharpe_Ratio(self, ret):
        return ret.mean() / ret.std() * np.sqrt(252)
    
    def Sortino_Ratio(self, ret):
        sigma_d = np.sqrt(1 / len(ret) * (np.where(ret<0, ret, 0) ** 2).sum())
        return ret.mean() / sigma_d * np.sqrt(252)
    
    def Information_Ratio(self, rets):
        diff = rets.Portfolio - rets.Benchmark
        return diff.mean() / diff.std()
    