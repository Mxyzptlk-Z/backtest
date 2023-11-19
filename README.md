# Frozen Backtest Framework
This is a strategy backtesting framework developed purely individually. It contains 5 major modules: Framework, AlphaBase, Util, Optimize and Client.
 
## 1. Frozen回测框架逻辑介绍

### 1.1 导入回测框架
```
from Framework import *
```
### 1.2 常规编写步骤

如图为Frozen系统主界面：

![系统主界面](https://files.mdnice.com/user/46794/4e120881-4675-477c-8a10-1379c136596b.png)

在terminal中运行Framework.py文件，等待数据获取/更新，待进度条完成，输出策略评价指标和净值曲线图。

![框架运行流程](https://files.mdnice.com/user/46794/c8b31000-5fa3-4f6c-b979-e620af50807a.png)

**系统运行逻辑及流程**：首先，初始化Client信息，将参数传入Framework；接着数据流通过api经过筛选及预处理后从Util中传入AlphaBase模块；在这里进行因子计算和打分后继续传入Framework进行选股；然后Framework将相关数据pass给Optimize，计算出权重后返回给Framework进行处理；最后根据策略回测收益调用Util的评价指标输出结果分析。

初始化框架参数如下：

```python
# Framework
index_code = 'tushare支持指数类型'
start_date = None
end_date = TODAY
max_stocks = 5
date_rule = 4
csv = True
html = True

# Util
PATH = './data cache'
token = 'tushare社区token'

# Client
init_capital = 1e6
slippage = 0.0001
commission = 0.0003
stamp_duty = 0.001
ub = np.Inf
lb = -0.08
slippage_type = 'byRate'
```

### 1.3 模块说明

下图显示了各模块之间的调用逻辑：

![模块调用逻辑弦图](https://files.mdnice.com/user/46794/074a1c72-8a91-4c31-aede-2cbd075a22ba.png)

### Framework

**Framework**是策略框架，起统筹规划的作用。它有两个任务：负责记录交易状态，计算每天的损益，生成交易日志，并根据持仓输出给出交易信号；进行绩效评估，绘制策略净值曲线，计算各种评价指标。

```python
Framework(index_code = '000905.SH', start_date = '20230101', end_date = '20230726', max_stocks = 5, date_rule = 4)
```

Invoke Frozen backtest framework and initialize data flow.

>  **Parameters:**
> - index_code (`str`) - String object denoting benchmark to trace
> - start_date (`str`, _optional_) - The strategy start date of format "YYYYmmdd"
> - end_date (`str`, _optional_) - The strategy end date of format "YYYYmmdd"
> - max_stocks (`int`, _defalut_: `5`) - Max number of stocks to hold, normally set larger than 3
> - date_rule (`int`, _defalut_: `4`) -  Day of week for position adjustment following the strategy weekday rule ["1"-"Mon", "2"-"Tue", "3"-"Wed", "4"-"Thu", "5"-"Fri"]
>
> **Returns:**
> - Frozen framework (`class`)

```python
Framework.result_analysis(csv = True, html = False)
```

Quantify strategy performance by creating charts and plotting relevant graphs.

>  **Parameters:**
> - csv (`boolean`, _default_: `True`) - If True, output .csv file including all stock positions
> - html (`boolean`, _default_: `False`) - If True, output .html file comprising strategy net value and all other evaluation indices

此外，Framework会自动生成交易日志，并对特殊的activities进行记录，除日常INFO记录持仓外，会有以下三个WARNING级别的记录：

- Portfolio reallocation day is holiday, rebalancing deferred to next trading day.
- Stop-loss point occurs for xxx on YYYY-mm-dd, closing long position.
- Dividend incident occurs for xxx on YYYY-mm-dd

### AlphaBase

**AlphaBase**是策略主体，也是整个策略的核心。所有的因子研究、因子合成都在这个模块里进行。

```python
def factor_mining(self):
    wms = pd.DataFrame({tic: ((data['hfq_high'].rolling(self.window).max() - data['hfq_close']) / (data['hfq_high'].rolling(self.window).max() - data['hfq_low'].rolling(self.window).min())) * 100 for tic, data in self.component.items()})
    roc = pd.DataFrame({tic: (data['hfq_close'] - data['hfq_close'].shift(self.window)) / data['hfq_close'].shift(self.window) for tic, data in self.component.items()})
  
    wms = ((wms - wms.rolling(self.timestamp).min()) / (wms.rolling(self.timestamp).max() - wms.rolling(self.timestamp).min()))[self.start_date:self.end_date]
    roc = ((roc - roc.rolling(self.timestamp).min()) / (roc.rolling(self.timestamp).max() - roc.rolling(self.timestamp).min()))[self.start_date:self.end_date]
  
    score = 0.5 * wms + 0.5 * roc
    return score
```

这里在计算因子值时采用的是pandas中强大的rolling方法，一行代码即可解决一个因子的计算；接着需要对因子进行标准化或归一化，注意在计算中不能引入未来函数；最后将所有因子合成，得到每只股票每天的因子得分。

### Optimize

**Optimize**是组合优化模块，起锦上添花的作用。它在一定程度上可以美化组合收益。该模块封装了Markowitz的Mean-Variance框架，apply了协方差矩阵估计的shrinkage方法。

这里的逻辑是，考虑到A股不能纯做空的特性（忽略融券做空），最大化目标函数。该问题的本质是凸优化问题，采用SQP算法数值解出最优权重，再输出到框架中。

$$ \underset {argmax \ \pmb{w}}{\max} \; \pmb{w^TR} - a \times \pmb{w^T} \pmb{\sum} \pmb{w} $$
$$ s.t.\ f(x)=\left\{\begin{aligned} w_i & \in (0,1) \\ \sum w_i & = 1 \end{aligned} \right. $$

其中协方差矩阵采用复杂度较低的portfolio of estimators进行估计。

### Util

**Util**是策略实用模块，起到辅助的作用。该类中存放了一些功能性函数，如数据获取/更新/存储/读取，以及日历调整和策略评价指标等。

### Client

**Client**是模拟经纪商模块，充当broker的作用。该类中存放了客户的基本信息，如本金/手续费/佣金等，此外还纳入了策略风控机制，包括滑点如何确定，止损函数设置，以及交易日志创建等内容。


## 2. 策略绩效分析

### 2.1 输出形式

框架共有以下五种回测结果输出，同时构建一个本地数据库：

（1）表格（tabulate）

在terminal中以表格形式打印策略评价指标。

![策略指标分析](https://files.mdnice.com/user/46794/41e64fc4-ecda-4f04-99db-934ae1ded45b.png)

从左到右分别为：理论净值，账户净值，基准净值，年化收益，beta，alpha，夏普比率，索提诺比率，信息比率（IR），胜率，最大回撤。

（2）图片（.png）

输出策略净值曲线，可自行选择是否保存为net_value.png。

![策略净值曲线](https://files.mdnice.com/user/46794/811a12f8-daef-4895-bb58-f0e1125eebb9.png)

如上，图中共有三条曲线，其中绿色线代表策略基准（跟踪）净值；蓝色线是策略理论净值，即不考虑交易成本和现实交易情况；橙色线是账户净值，为上费（手续费，佣金，印花税等）后的现实净值。

（3）日志（.log）

自动在cwd中创建logs文件夹，其中包含“index_code + account_activities.log”文件，如下为日志示例：

![交易日志](https://files.mdnice.com/user/46794/b8c767dc-6590-45d1-b8b7-47f4be18bfd0.png)

目前框架对于当周的处理只记录买入，后续会加入跟踪（tracing）的功能。

（4）Excel（.csv）

若所设置的`csv = True`，则在cwd中自动创建stock_orders.csv文件。里面记录了每周的持仓数据（时间、股票代码、权重），按照时间顺序排列，最新持仓即为当周换仓需求，一般根据这个进行人工下单。

（5）网页（.html）

若所设置的`html = True`，则在cwd中自动创建tearsheet.html文件。该网页由[quantstats](https://github.com/ranaroussi/quantstats)包自动生成，其中包含了各种策略评价指标以及绩效评估图表。

（6）文件夹（data_cache）

构建本地数据库，在cwd中自动创建index_code文件夹，数据存储为.pkl格式。

- index.pkl - 基准指数daily量价数据

- component.pkl - 指数成分股daily量价数据

- acct.pkl - 指数成分股daily会计数据

- dividend.pkl - 指数成分股分红详情

### 2.2 指标计算

下面对投资策略回测图中显示的策略风险评价指标计算公式作相关说明：

- 绝对收益率（Absolute Returns）：是策略在回测期间内取得的收益，其计算公式如下：

$$ AbsoluteReturn = \frac{P_{end} - P_{start}}{P_{start}} $$

- 累计收益率（Cumulative Returns）:

$$ CumulativeReturn = \prod_{i=1}^{n}\, (1+p_i) - 1 $$

- 年化收益率（Annualized Returns）：是策略投资期限为一年所获得的收益，其计算公式如下：

$$ AnnualizedReturn = (\frac{P_{end}}{P_{start}})^{252/n} - 1 $$

- 相对收益率（Relative Returns）：是策略相对于业绩比较基准的收益，其计算公式如下：

$$ RelativeReturn = R_p - R_m $$

- Alpha值：Alpha量化了交易者从市场中获得的与市场走势无关的交易回报，其计算公式如下：

$$ Alpha = R_p - [R_f + \beta (R_m - R_f)] $$

- Beta值：Bata量化了投资的系统风险，其计算公式如下：

$$ \beta = \frac {Cov(R_{p_t}, R_{m_t})}{\sigma_{m_t}^2} $$

- 收益波动率（Volatility）：主要用于测量资产的风险，波动率越高，风险越大，其计算公式如下：

$$ \sigma_p = \sqrt{\frac{252}{n-1} \sum_{i=1}^n(p_i-\bar{p})^2} $$

- 夏普比率（Sharpe）：Sharpe代表交易者每多承担一份风险，相应的就可以拿到几份超额收益，其计算公式如下：

$$ Sharpe = \frac{R_p - R_f}{\sigma_p} $$

- 信息比率（Information Ratio）：用于衡量单位超额风险带来的超额收益，其计算公式如下：

$$ InformationRatio = \frac{R_p - R_m}{\sigma_t^{p-m}} $$

- 索提诺比率（Sortino Ratio）：与夏普比率类似, 不同的是在计算波动率时它采用的是下行标准差，可以看作是Sharpe的一种修正方式，其计算公式如下：

$$ SortinoRatio = \frac{R_p - R_f}{\sigma_d} $$

$$ \sigma_d = \sqrt{\frac{252}{n-1} \sum_{i=1}^n [min(p_i-R_f,0)]^2} $$

- 胜率（Win Rate）：交易盈利次数占总交易次数的比例，其计算公式如下：

$$ WinRate = \frac{N_{win}}{N_{total}} $$

- 最大回撤（Max Drawdown）：用于衡量策略可能出现的最糟糕的情况，其计算公式如下：

$$ MaxDrawdown_t = max(1 - \frac{P_j}{P_i})\ , \quad t \geq j \geq i $$


## 3. 回测细节

### 3.1 回测时间说明

#### 3.1.1. 策略调仓时间

我们的回测框架中，调仓周期默认设置为**周频**，且根据历史回测来看，在每周四进行换仓取得的效果最好，因此策略默认每周四按照当天开盘价进行定期调仓。

#### 3.1.2. 调仓周期设置

回测框架通过[pandas.DataFrame.resample](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html#pandas.DataFrame.resample)函数进行调仓周期设置。

```python
weekly_score = score.resample(freq = self.date_rule).mean()
```

参数完全继承resample函数的设置，`freq`取值有'B'（日）、'W'（周）、'M'（月）等，还可以设置时间前后偏移量`offset`，所以该方法适用于所有月、周、日、小时调仓，而且调仓周期的设置也更为灵活和多样。

以下是`freq`的一部分可取范围：

| Alias | Description |
| :--: | :--: |
|  B  |  business day frequency  |
|  D  |  calendar day frequency  |
|  W  |  weekly frequency  |
|  M  |  month end frequency  |
|  SM  |  semi-month end frequency (15th and end of month)  |
|  Q  |  quarter end frequency  |
|  A, Y  |  year end frequency  |
|  H  |  hourly frequency  |
|  T, min  |  minutely frequency  |
|  S  |  secondly frequency  |

### 3.2 下单（调仓）

### 3.2.1. 下单方式

方式一：人工手动下单

方式二：QMT或Ptrade进行自动下单

### 3.2.2. 调仓方式

方式一：按买卖证券的数量进行调仓（order），涉及下单函数中的`volume`参数

方式二：按指定金额调仓（order_value），涉及下单函数中的`value`参数

方式三：按总资产百分比调仓（order_percent），涉及下单函数中的`percent`参数

### 3.2.3. 仓位限制

由于A股市场的特性，在进行交易时，若根据Optimize给出的权重算得的持仓数量小于1手（100股），则不进行买入，并从持仓中剔除，因此：

- 策略理论净值曲线：假设资金使用率为100%

- 现实账户净值曲线：应预留足够的现金以cover交易费用（手续费及税费），所以一般资金无法全部投入使用

### 3.2.4. 委托价格

通过设置下单函数中的`price`参数设置委托价格，再结合设置的滑点，确定最终的成交价。

委托价格price既可以自行设定价格，也可以将下单日的'open'（开盘价），'vwap'（均价），或前一交易日的'close'（收盘价）作为委托价格，所以委托价格共有4种设置方式，能很好地满足各种需求。

回测期间默认以上一交易日收盘价作为成交价结算，但由于可能存在持续持仓的情况，且实盘中买卖操作均于当日执行，因此采用当日开盘价作为委托价更加合理（to be discussed）。

### 3.2.5. 买卖方向

一般通过下单函数中的`trade_side`参数来设置买卖方向，买卖方向共有2种：'buy'：开多、'short'：开空。

但由于框架只支持纯多头策略，所以卖卖方向仅限'buy'。

### 3.2.6. 涨跌停

对于停牌或是跌停的股票，回测系统会对其自动删除，不进行买卖交易；涨停股票仍可以涨停价买入。

### 3.2.7. 退市

在回测过程中，若持仓中出现退市股票，回测系统会将退市股票按**最后一个交易日的收盘价**作为退市价格进行清仓处理。

### 3.3 持仓期间调整

### 3.3.1. 滑点

通过Client中的apply_slippage对象属性来设置滑点：

```python
def apply_slippage(self, price, pct_slippage, slippage_type):
    if slippage_type == 'byRate':
        price_with_slippage = price * (1 + pct_slippage)
    elif slippage_type == 'byVolume':
        price_with_slippage = price + pct_slippage
    else:
        print('Please indicate which slippage type.')
    return price_with_slippage
```

共有两种滑点设置类型：`byRate`和`byVolume`

- 百分比（`byRate`）：如设置的`slippage_type = 'byRate'`，`pct_slippage = 0.0001`，买卖交易时的成交价为委托价格加上委托价的0.01%

- 固定金额（`byVolume`）: 如设置的`slippage_type = 'byVolume'`，`pct_slippage = 0.02`，买卖交易时的成交价为委托价格加上0.02

Frozen框架默认按照byRate 1bp进行设置，比较符合股票市场的惯例。

### 3.3.2. 拆分合并与分红

当股票发生拆分，合并或是分红配股时，股票价格会发生较大的变动，使得当前价格变得不连续，这时我们会对价格进行复权来保持价格的连续性，所以回测过程中提取历史行情数据时，可以选择是否采用定点复权来保持价格的连续性。

在下单交易时，为保证回测的真实性，设置的委托价使用的仍是真实的行情数据，没有经过复权处理。

定点复权具体操作：定点复权一般是对定点之前的价格进行前复权处理，对定点之后的价格保持不变，若定点之后遇到除权除息日，则进行后复权处理。

一般来说，使用后复权价计算股票收益率，从而获得策略理论净值；此外，后复权价还被用来计算策略胜率（未包含当周），注意要考虑滑点和手续费。而真实价格（不复权）则用于交易，得到现实账户净值。

在策略持有期间，若持仓股票出现分红送股的情况，需要对持股数量和可用现金进行调整：

```python
num_shares *=（1 + stk_div）
avail_cash += cash_div
```

### 3.3.3. 交易税费

可通过Client中的`commission`参数来设置交易费用，默认情况为0.0003，双边征收；通过`stamp_duty`设置税费，默认为0.001，向卖方单边征收。

- 买入：仅需收取手续费

- 卖出：该费用是佣金和税费的综合费用，为手续费+印花税

连续持仓的手续费减免调整：若连续两周具有相同持仓或持仓交叉，则在调仓日不需要进行重复买卖，可以省去一部分手续费的支出。

### 3.3.4. 止盈止损

回测框架分别设置了对价格（price）和对收益率（return）的止盈和止损函数，从而进行风险控制。

以对收益率的止盈止损函数为例：

```python
def pnl_lim(self, ret):
    PROFIT_LIMIT = np.array(np.where(ret>=self.upper_bound))
    LOSS_LIMIT = np.array(np.where(ret<=self.lower_bound))
    ret_filter = ret.copy()
    for idx in np.hstack([LOSS_LIMIT, PROFIT_LIMIT]).T:
        ret_filter.iloc[idx[0]+1:,idx[1]] = 0
    return ret_filter
```

策略默认对于单只股票无盈利限制（`ub = np.Inf`），而亏损限制为-80%（`lb = -0.8`），整体亏损限制为-20%（`total_loss = -0.2`）。

