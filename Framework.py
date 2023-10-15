import pandas as pd
import numpy as np
import datetime
import warnings
import logging
import quantstats as qs
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm
from AlphaBase import AlphaBase
from Optimize import PortOpt
warnings.filterwarnings('ignore')


class Framework(AlphaBase, PortOpt):
    
    def __init__(self, index_code, start_date, end_date, max_stocks, date_rule, freq):
        """
        This method initializes the parameters entering the framework.
        """
        AlphaBase.__init__(self, index_code, start_date, end_date)
        PortOpt.__init__(self, index_code, start_date, end_date, max_stocks)
        self.date_rule = date_rule
        self.freq = freq
    
    def __repr__(self):
        """
        This method returns a string representation of a Framework object.
        """
        return 'Backtest_Framework, index_code=%s, start_date=%s, end_date=%s, max_stocks=%s, date_rule=%s' % (self.index_code, self.start_date, self.end_date, self.max_stocks, self.date_rule)
    
    def port_performance(self):
        """
        This method is the primary backtest framework cerebro.
        """
        score, index, component, dividend = self.factor_mining()
        timedelta = self.period_delta(self.freq)
        ONE_DAY = datetime.timedelta(days=1)
        self.log()
        
        # adjusted price for dividend
        real_open = pd.DataFrame({tic: data['open'] for tic, data in component.items()}).fillna(method='bfill')
        real_close = pd.DataFrame({tic: data['close'] for tic, data in component.items()}).fillna(method='bfill')
        hfq_open = pd.DataFrame({tic: data['hfq_open'] for tic, data in component.items()}).fillna(method='bfill')
        hfq_close = pd.DataFrame({tic: data['hfq_close'] for tic, data in component.items()}).fillna(method='bfill')

        rets = hfq_close.pct_change()
        mkt_rets = index['close'].pct_change()
        period_score = score.resample(self.date_rule).mean()
        
        order_target = []
        for i in range(len(period_score)):
            order_target.append(period_score.iloc[i,:].sort_values(ascending=False).index[0:self.max_stocks] if not period_score.iloc[i,:].dropna().empty else order_target[i-1])
        
        last_day_value = self.init_capital
        last_period_value = self.init_capital
        port_rets = np.array(())
        account_rets = []
        dates = []
        w = np.zeros((len(order_target), self.max_stocks))
        total_order = (len(order_target) - 1) * self.max_stocks
        win = 0
        
        for i in tqdm(range(len(order_target))):
                
            holdings = order_target[i]
            calc_date = period_score.index[i]
            shift_date = self.next_trade_day(calc_date)
            period_start = calc_date + ONE_DAY
            period_end = calc_date + timedelta if i != len(order_target) - 2 else self.end_date

            target_rets = rets[holdings]
            target_real_open = real_open[holdings]
            target_real_close = real_close[holdings]
            target_hfq_open = hfq_open[holdings]
            target_hfq_close = hfq_close[holdings]

            out_of_sample = target_rets[period_start : period_end]
            if out_of_sample.empty:
                logging.info(calc_date + ONE_DAY)
                logging.info('Whole period is holiday.')
                continue

            # end_date isn't last period shift date
            if i == len(order_target) - 1 and calc_date > datetime.datetime.strptime(self.end_date,'%Y%m%d'):
                print('Warnings: Not coincide with date_rule for the last holding period.')
                break

            if self.exist_holiday_except_weekend(calc_date, shift_date):
                logging.info('Portfolio reallocation day is holiday, rebalancing deferred to next trading day.')

            logging.info(shift_date)
            logging.info('Placing orders (long stocks):\n%s', holdings.tolist())

            # for covariance matrix estimation
            in_sample = target_rets[calc_date - datetime.timedelta(days=90) : calc_date]
            mkt_in_sample = mkt_rets[calc_date - datetime.timedelta(days=90) : calc_date]
            w[i] = self.res(in_sample, mkt_in_sample)

            # deal with last period shift date
            if i == len(order_target) - 1 and calc_date == datetime.datetime.strptime(self.end_date,'%Y%m%d'):
                break
            
            # if wish to change to other (monthly, quarterly or yearly) frequency, change params in relativedelta
            period_real_close = target_real_close[period_start : period_end]
            period_hfq_close = target_hfq_close[period_start : period_end]
            real_order_price = target_real_open.loc[shift_date]
            hfq_order_price = target_hfq_open.loc[shift_date]

            # due to suspend incidents, some order prices might be missing 
            num_shares = ((last_period_value * w[i] / real_order_price) / 100).astype('int')
            # the characteristic of A share, if num_shares < 1, dump the stock
            num_shares[num_shares<1] = 0
            num_shares *= 100
            logging.info('Number of shares respectively:\n%s', np.array(num_shares))

            # apply slippage
            price_with_slippage = self.apply_slippage(real_order_price, self.slippage, slippage_type='byRate')
            begin_total_position = (num_shares * price_with_slippage).sum()

            # calculate commission
            # adjust commission for continuous holding
            if i != 0:
                last_period_holdings = order_target[i-1]
                ct_hold = list(set(holdings).intersection(set(last_period_holdings)))
                indicator = list(map(lambda x: x not in ct_hold, holdings))
                buy_commission = (num_shares * price_with_slippage * indicator).sum() * self.commission
            else:
                buy_commission = begin_total_position * self.commission
            logging.info('buy commission cost: %.2f', buy_commission)

            # calculate transaction cost
            avail_cash = max(last_period_value - begin_total_position - buy_commission, 0)
            logging.info('available cash: %.2f', avail_cash)

            hfq_position_cost = self.apply_slippage(hfq_order_price, self.slippage, slippage_type='byRate') * (1 + self.commission)

            for date in out_of_sample.index:
                for j in range(len(out_of_sample.columns)):
                    stock = out_of_sample.columns[j]
                    info = dividend[stock]
                    if date in info['ex_date'].tolist():
                        # sometimes there exists multiple indices for a single date
                        # first cash dividend then stock dividend
                        avail_cash += np.array(info[info['ex_date']==date]['cash_div'])[0] * num_shares[j]
                        num_shares[j] *= (1 + np.array(info[info['ex_date']==date]['stk_div'])[0])
                        logging.info('dividend incident occurs for %s on %s', stock, date)
                        logging.info('rebalancing account holdings:\n%s', np.array(num_shares))
                    else:
                        pass
                    
                    if out_of_sample.loc[date, stock] <= self.lower_bound:
                        avail_cash += num_shares[j] * period_real_close.loc[date, stock]
                        # deals with win rate at stop-loss point
                        if period_hfq_close.loc[date, stock] > hfq_position_cost[j]:
                            win += 1
                        num_shares[j] = 0
                        logging.info('stop-loss point occurs for %s on %s', stock, date)
                        logging.info('closing long position:\n%s', np.array(num_shares))
                    elif out_of_sample.loc[date, stock] >= self.upper_bound:
                        avail_cash += num_shares[j] * period_real_close.loc[date, stock]
                        if period_hfq_close.loc[date, stock] > hfq_position_cost[j]:
                            win += 1
                        num_shares[j] = 0
                        logging.info('stop-profit point occurs for %s on %s', stock, date)
                        logging.info('closing long position:\n%s', np.array(num_shares))
                    else:
                        pass
                
                dates.append(date)
                if date != out_of_sample.index[-1]:
                    daily_account_value = (num_shares * period_real_close.loc[date]).sum() + avail_cash
                    daily_pnl = daily_account_value / last_day_value - 1
                    account_rets.append(daily_pnl)
                    last_day_value = daily_account_value
                else:
                    # if last day of holding period, deal with sell commission
                    end_total_position = (num_shares * period_real_close.loc[date]).sum()
                    sell_commission = end_total_position * (self.commission + self.stamp_duty)
                    logging.info('sell commission cost: %.2f', sell_commission)
                    daily_account_value = end_total_position + avail_cash - sell_commission
                    daily_pnl = daily_account_value / last_day_value - 1
                    account_rets.append(daily_pnl)
                    last_day_value = daily_account_value
            period_account_value = end_total_position + avail_cash - sell_commission
            period_pnl = period_account_value - last_period_value
            last_period_value = period_account_value
            logging.info('period pnl: %.2f\n', period_pnl)
            win += (period_hfq_close.loc[date] > hfq_position_cost).sum()
            
            # Theoretical daily return
            # Set loss/profit limit for strategy, replace the out_of_sample return after the certain date by 0
            filter_ret = self.pnl_lim(out_of_sample)
            port_rets = np.append(port_rets, np.dot(filter_ret, w[i].T))
        
        total_pnl = period_account_value - self.init_capital
        logging.info('total pnl: %.2f\n\n', total_pnl)
        port_ret = pd.DataFrame({'Portfolio': port_rets, 'Account': account_rets, 'Benchmark': mkt_rets[dates]}, index = dates)
        win_rate = win / total_order

        # Retrieve order information
        order_history = pd.DataFrame()
        adj_date = self.next_trade_dates(period_score.index)
        for order, weight, date in zip(order_target, w, adj_date):
            period_order = pd.DataFrame()
            period_order['date'] = [date] * len(order)
            period_order['stock'] = order
            period_order['weight'] = weight
            order_history = pd.concat([order_history, period_order])
        
        order_history.set_index(['date', 'stock'], inplace=True)
    
        return port_ret, order_history, win_rate
    
    def result_analysis(self, csv = False, html = False):
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
        
        if csv is True:
            order_history.to_csv('stock orders.csv')

        if html is True:
            qs.reports.html(port_ret['Account'], benchmark=port_ret['Benchmark'], output='stats.html', title='Portfolio Performance')
        
        print(tabulate(table, headers='keys', tablefmt='grid'))
    
    
if __name__ == '__main__':
    
    alpha = Framework('000905.SH', '20230101', '20231014', 5, 'W-WED', 'W')
    alpha.result_analysis(csv = False, html = False)
    
    