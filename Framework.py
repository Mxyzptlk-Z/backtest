import pandas as pd
import numpy as np
import datetime
import warnings
import logging
import scipy.optimize as sco
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tabulate import tabulate
from dateutil.relativedelta import relativedelta
from AlphaBase import AlphaBase
warnings.filterwarnings('ignore')


class Framework(AlphaBase):
    
    def __init__(self, index_code, start_date, end_date, max_stocks, date_rule):
        """
        This method initializes the parameters entering the framework.
        """
        super().__init__(index_code, start_date, end_date, max_stocks, date_rule)
    
    def __repr__(self):
        """
        This method returns a string representation of a Framework object.
        """
        return 'Backtest_Framework, index_code=%s, start_date=%s, end_date=%s, max_stocks=%s, week_rule=%s' % (self.index_code, self.start_date, self.end_date, self.max_stocks, self.date_rule)
    
    def port_performance(self):
        """
        This method is the primary backtest framework cerebro.
        """
        score, index, component, dividend = self.factor_mining()
        # score, index, component, dividend = self.apply_alpha()
        self.log()
        
        # adjusted price for dividend
        real_close = pd.DataFrame({tic: data['close'] for tic, data in component.items()}).fillna(method = 'bfill')
        hfq_close = pd.DataFrame({tic: data['hfq_close'] for tic, data in component.items()}).fillna(method = 'bfill')
        rets = hfq_close.pct_change()
        mkt_rets = index['close'].pct_change()
        weekly_score = score.resample(self.date_rule).mean()
        
        order_target = []
        for i in range(len(weekly_score)):
            order_target.append(weekly_score.iloc[i,:].sort_values(ascending=False)[0:self.max_stocks].index)
        
        # Portfolio optimization
        
        def gen_single_index(stock, market):
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
        
        cons = ({'type':'eq','fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for x in range(self.max_stocks))
        eweights = np.array(self.max_stocks * [1./self.max_stocks])
        
        # total_capital = self.init_capital
        # last_total_capital = self.init_capital
        last_day_value = self.init_capital
        last_week_value = self.init_capital
        port_rets = np.array(())
        account_rets = []
        dates = []
        w = np.zeros((len(order_target)-1, self.max_stocks))
        
        for i in range(len(order_target) - 1):

            logging.info(self.next_business_day([weekly_score.index[i]])[0])
            logging.info('Placing orders (long stocks):\n%s', order_target[i].tolist())

            target_rets = rets[order_target[i]]
            real_price = real_close[order_target[i]]
            
            # Covariance matrix estimation
            in_sample = target_rets[weekly_score.index[i] - datetime.timedelta(days=90) : weekly_score.index[i]]
            mkt_in_sample = mkt_rets[weekly_score.index[i] - datetime.timedelta(days=90) : weekly_score.index[i]]
            
            cov = 1/3 * in_sample.cov() + 1/3 * np.diag(in_sample.var().tolist()) + 1/3 * gen_single_index(in_sample, mkt_in_sample)
            
            def min_func_sharpe(weights):
                return - np.sum(in_sample.mean() * weights) * 252 + 20 * np.sqrt(np.dot(weights.T, np.dot(cov * 252, weights)))
            
            opts = sco.minimize(min_func_sharpe, eweights, method='SLSQP', bounds=bnds, constraints=cons)
            w[i] = opts['x']
            
            # resample by week so +7
            # out_of_sample = target_rets[weekly_score.index[i] + datetime.timedelta(days=1) : weekly_score.index[i] + datetime.timedelta(days=7)]
            # ext_real_price = real_price[weekly_score.index[i] : weekly_score.index[i] + datetime.timedelta(days=7)]
            # if wish to change to other (monthly, quarterly or yearly) frequency, change params in relativedelta
            out_of_sample = target_rets[weekly_score.index[i] + datetime.timedelta(days=1) : weekly_score.index[i] + relativedelta(weeks=1)]
            ext_real_price = real_price[weekly_score.index[i] : weekly_score.index[i] + relativedelta(weeks=1)]
            
            # num_shares = ((total_capital * w[i] / ext_real_price.iloc[0]) / 100).astype('int')
            num_shares = ((last_week_value * w[i] / ext_real_price.iloc[0]) / 100).astype('int')
            # due to the characteristic of A share, if num_shares < 1, dump the stock
            num_shares[num_shares<1] = 0
            num_shares *= 100
            logging.info('Number of shares respectively:\n%s', np.array(num_shares))
            # apply slippage
            # slippage_cost = (num_shares * ext_real_price.iloc[0]).sum() * slippage
            price_with_slippage = self.apply_slippage(ext_real_price.iloc[0], self.slippage)
            total_position = (num_shares * price_with_slippage).sum()
            # calculate commission
            commission_cost = total_position * self.commission
            logging.info('commission cost: %.2f', commission_cost)
            # calculate transaction cost
            # transaction_cost = slippage_cost + commission_cost
            # avail_cash = max(total_capital - total_position - commission_cost, 0)
            avail_cash = max(last_week_value - total_position - commission_cost, 0)
            logging.info('available cash: %.2f', avail_cash)

            for date in out_of_sample.index:
                for j in range(len(out_of_sample.columns)):
                    stock = out_of_sample.columns[j]
                    info = dividend[stock]
                    if date in info['ex_date'].tolist():
                        num_shares[j] *= (1 + float(info[info['ex_date']==date]['stk_div']))
                        avail_cash += float(info[info['ex_date']==date]['cash_div'])
                        logging.info('dividend incident occurs for %s on %s', stock, date)
                        logging.info('rebalancing account holdings:\n%s', np.array(num_shares))
                    else:
                        pass
                    
                    if out_of_sample.loc[date, stock] <= self.lower_bound:
                        avail_cash += num_shares[j] * ext_real_price.loc[date, stock]
                        num_shares[j] = 0
                        logging.info('stop-loss point occurs for %s on %s', stock, date)
                        logging.info('closing long position:\n%s', np.array(num_shares))
                    elif out_of_sample.loc[date, stock] >= self.upper_bound:
                        avail_cash += num_shares[i] * ext_real_price.loc[date, stock]
                        num_shares[j] = 0
                        logging.info('stop-profit point occurs for %s on %s', stock, date)
                        logging.info('closing long position:\n%s', np.array(num_shares))
                    else:
                        pass

                daily_account_value = (num_shares * ext_real_price.loc[date]).sum() + avail_cash
                daily_pnl = daily_account_value / last_day_value - 1
                account_rets.append(daily_pnl)
                last_day_value = daily_account_value
                dates.append(date)
            weekly_account_value = (num_shares * ext_real_price.loc[date]).sum() + avail_cash
            weekly_pnl = weekly_account_value - last_week_value
            last_week_value = weekly_account_value
            logging.info('weekly pnl: %.2f\n', weekly_pnl)

            # # Account daily return
            # daily_holding_value = num_shares * ext_real_price.iloc[0]  # bin shift day
            # share_weight = daily_holding_value / daily_holding_value.sum()
            # ex_cost_ret = out_of_sample.copy()
            # if ex_cost_ret.empty:
            #     pass
            # else:
            #     ex_cost_ret.iloc[0] = (ext_real_price.iloc[1] - price_with_slippage) / price_with_slippage - self.commission
            # ex_cost_ret = self.pnl_lim(ex_cost_ret)
            # account_rets = account_rets.append((share_weight * ex_cost_ret).sum(axis=1))

            # # Account weekly PnL
            # filter_price = self.loss_lim(ext_real_price, out_of_sample)
            # holding_period_value = (num_shares * filter_price.iloc[-1]).sum()
            # total_capital = holding_period_value + avail_cash
            # weekly_pnl = total_capital - last_total_capital
            # logging.info('weekly pnl: %.2f\n', weekly_pnl)
            # last_total_capital = total_capital
            
            # Theoretical daily return
            # Set loss/profit limit for strategy, replace the out_of_sample return after the certain date by 0
            filter_ret = self.pnl_lim(out_of_sample)
            port_rets = np.append(port_rets, np.dot(filter_ret, w[i].T))
        
        # total_pnl = total_capital - self.init_capital  # it's larger because it includes account available cash
        # logging.info('total pnl: %.2f\n\n', total_pnl)
        total_pnl = weekly_account_value - self.init_capital
        logging.info('total pnl: %.2f\n\n', total_pnl)
        port_ret = pd.DataFrame({'Portfolio': port_rets, 'Account': account_rets, 'Benchmark': mkt_rets[dates]}, index = dates)
        
        # win rate
        weekly_ret = hfq_close.resample(self.date_rule).last() / hfq_close.resample(self.date_rule).last().shift(1) - 1
        
        total_order = (len(order_target) - 1) * 5
        win = 0
        for i in range(len(order_target)-1):
            num = (weekly_ret.loc[weekly_score.index[i+1], order_target[i]] > 0).sum()
            win += num
        win_rate = win / total_order
        
        # Retrieve order information
        order_history = pd.DataFrame()
        adj_date = self.next_business_day(weekly_score.index)
        for order, weight, date in zip(order_target, w, adj_date):
            weekly_order = pd.DataFrame()
            weekly_order['date'] = [date] * len(order)
            weekly_order['stock'] = order
            weekly_order['weight'] = weight
            order_history = order_history.append(weekly_order)
        order_history.set_index(['date','stock'], inplace=True)
    
        return port_ret, order_history, win_rate
    
    def result_analysis(self):
        """
        This method generates plots and major assessment indicators of a strategy..
        """
        port_ret, order_history, win_rate = self.port_performance()
        
        # Plot
        net_value = (1 + port_ret).cumprod()
        net_value.plot(figsize=(10, 6))
        plt.title('Portfolio Performance')
        plt.xlabel('trade date')
        plt.ylabel('net value')
        plt.show()
        
        final_nv = net_value[-1:].iloc[0,]
        ar = self.annual_rate(port_ret['Portfolio'])
        beta = self.Beta(port_ret)
        alpha = self.Alpha(port_ret)
        sharpe = self.Sharpe_Ratio(port_ret['Portfolio'])
        sortino = self.Sortino_Ratio(port_ret['Portfolio'])
        information = self.Information_Ratio(port_ret)
        max_drawdown = self.Max_Drawdown(net_value['Portfolio'])
        
        table = {'Theory': [round(final_nv['Portfolio'], 4)],
                 'Account': [round(final_nv['Account'], 4)],
                 'Benchmark': [round(final_nv['Benchmark'], 4)],
                 'Annual Rate': [round(ar, 4)],
                 'Beta': [round(beta, 4)],
                 'Alpha': [round(alpha, 4)],
                 'Sharpe': [round(sharpe, 4)],
                 'Sortino': [round(sortino, 4)],
                 'IR': [round(information, 4)],
                 'Win Rate': [round(win_rate, 4)],
                 'Max Drawdown': [round(max_drawdown, 4)]}
        
        # order_history.to_csv('stock orders.csv')
        print(tabulate(table, headers='keys', tablefmt='grid'))
    
    
if __name__ == '__main__':
    
    alpha = Framework('399300.SZ', '20230101', '20230424', 5, 4)
    alpha.result_analysis()
    
    