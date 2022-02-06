# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 11:12:05 2021

@author: Hatricano
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from pandas import DatetimeIndex
from datetime import timedelta 
from datetime import datetime

import statsmodels.api as sm
from statsmodels.formula.api import ols
from patsy import dmatrices
import math

final_output = pd.DataFrame(columns = ['Equal Weighted','Value Weighted'], index = ['Capm Alpha','Capm Alpha t-stat', '3 Factor FF Alpha','3 Factor Alpha t-stat', 'Sharpe Ratio']).fillna(0.0)
final_output_exc500 = pd.DataFrame(columns = ['Equal Weighted','Value Weighted'], index = ['Capm Alpha','Capm Alpha t-stat', '3 Factor FF Alpha','3 Factor Alpha t-stat', 'Sharpe Ratio']).fillna(0.0)

# Daily Returns
daily_returns = pd.read_csv('CRSP_Daily.csv')
daily_returns.date = daily_returns.date.astype(str)
daily_returns.date = pd.DatetimeIndex(daily_returns.date)
daily_returns = daily_returns[~daily_returns.RET.isin(['B','C'])]
daily_returns = daily_returns[daily_returns.SHRCD.isin([10,11])]
daily_returns = daily_returns[~daily_returns.SICCD.isin(list(range(6000,7000)))]
daily_returns = daily_returns[daily_returns.CFACPR>0]
daily_returns = daily_returns[daily_returns.PRC>0]
daily_returns = daily_returns[daily_returns.SHROUT>0]

# FamaFrench Returns
ff_returns = pd.read_csv('DailyFF.csv')
ff_returns.Date = ff_returns.Date.astype(str)
ff_returns.Date = pd.DatetimeIndex(ff_returns.Date)
ff_returns  = ff_returns[ff_returns.Date>='1980-01-01']
ff_returns  = ff_returns[ff_returns.Date<='2020-12-31']
ff_returns['Mkt_RF'] = ff_returns['Mkt_RF']/100
ff_returns['SMB'] = ff_returns['SMB']/100
ff_returns['HML'] = ff_returns['HML']/100
ff_returns['RF'] = ff_returns['RF']/100


# Stock Splits
stock_splits = pd.read_csv('StockSplits.csv')
stock_splits = stock_splits[stock_splits['DISTCD']==5523]
stock_splits = stock_splits[stock_splits['FACPR']>=0]
stock_splits = stock_splits[stock_splits['FACPR']==stock_splits['FACSHR']]
stock_splits = stock_splits[stock_splits['FACSHR']>=1]
stock_splits['final_date'] = 0

for i in stock_splits.index.values:
    if stock_splits.loc[i].DCLRDT>0:
        stock_splits.loc[i,'final_date'] = stock_splits.loc[i].DCLRDT
        
    if stock_splits.loc[i].RCRDDT>0:
        stock_splits.loc[i,'final_date'] = stock_splits.loc[i].PAYDT
        
    if stock_splits.loc[i].PAYDT>0:
        stock_splits.loc[i,'final_date'] = stock_splits.loc[i].RCRDDT

    if stock_splits.loc[i].EXDT>0:
        stock_splits.loc[i,'final_date'] = stock_splits.loc[i].EXDT

stock_splits.final_date = stock_splits.final_date.astype(int)
stock_splits.final_date = stock_splits.final_date.astype(str)
stock_splits.final_date = pd.DatetimeIndex(stock_splits.final_date)
stock_splits = stock_splits[['PERMNO', 'DISTCD', 'FACPR', 'FACSHR', 'final_date']]


#### Part 1 
### Comparying return on day of announcement and next day vs market.

# Same Day
d1 = pd.merge(stock_splits, daily_returns[['PERMNO','date', 'RET' ]], left_on= ['PERMNO','final_date'],right_on=['PERMNO','date'])
d1 = d1[['PERMNO', 'DISTCD', 'FACPR', 'FACSHR', 'final_date', 'RET']]

# Next Day
d2 = stock_splits.copy()
d2['final_date'] = d2['final_date']+pd.DateOffset(1)
d2 = pd.merge(d2, daily_returns[['PERMNO','date', 'RET' ]], left_on= ['PERMNO','final_date'],right_on=['PERMNO','date'])
d2 = d2[['PERMNO', 'DISTCD', 'FACPR', 'FACSHR', 'final_date', 'RET']]

final = pd.concat([d1,d2])
final.dropna(inplace=True)
final = pd.merge(final, ff_returns[['Date','Mkt_RF']], left_on = ['final_date'],right_on = ['Date'])
final['RET'] = final['RET'].astype(float)
final['Mkt_RF'] = final['Mkt_RF'].astype(float)
final['excess_return'] = final['RET'].astype(float) - final['Mkt_RF'].astype(float)


# Avg.return
print('Average daily return on stock split is ', round(final.RET.astype(float).mean() * 100, 2 ))
print('Average excess daily return over market on stock split is ', round(final.excess_return.mean()*100,2))
final.plot(x='FACPR', y='RET', label='Returns vs FACPR', kind = 'scatter')
final.plot(x='FACSHR', y='RET', label='Returns vs FACSHR', kind = 'scatter')

#### Part 2
### Building Portfolio

dates = pd.to_datetime(ff_returns[ff_returns.Date>'1980-01-01'].Date.values)
dates_minus_6months = dates + pd.DateOffset(-180)

stocks_to_consider={}

for date in range(len(dates)):
    temp = stock_splits[['PERMNO','final_date']][stock_splits.final_date <= dates[date]]
    temp = temp[temp.final_date >= dates_minus_6months[date]]
    stocks_to_consider[dates[date]] = temp.PERMNO.values

stock_split_returns = daily_returns[daily_returns.PERMNO.isin(stock_splits.PERMNO)]
stock_split_returns = stock_split_returns[stock_split_returns.date>='1980-01-01']

stock_split_returns_v2 = stock_split_returns[['PERMNO', 'RET', 'date']]
stock_split_returns_v2.RET = stock_split_returns_v2.RET.astype(float)
stock_split_returns_v2 = pd.pivot_table(data = stock_split_returns_v2, columns = 'PERMNO', values='RET', index = 'date').fillna(0.0)


## Equal Weights

portfolio_return = pd.DataFrame(columns = ['Equal_weight_returns'], index = stock_split_returns_v2.index).fillna(0.0)

for i in portfolio_return.index:
    portfolio_return.loc[i,'Equal_weight_returns'] = sum(stock_split_returns_v2.loc[i])/len(stock_split_returns_v2.loc[i]>0)

portfolio_return.reset_index()

ff_market_reg = pd.merge(left = ff_returns, right = portfolio_return.reset_index(), left_on = ['Date'], right_on = ['date'] ).fillna(1.0)[['Date','Mkt_RF','Equal_weight_returns','RF','SMB','HML']]
ff_market_reg['Equal_weight_returns_minus_RF'] = ff_market_reg['Equal_weight_returns'] - ff_market_reg['RF']

# Capm
y, X = dmatrices('Equal_weight_returns_minus_RF ~ Mkt_RF', data=ff_market_reg, return_type='dataframe')
capm_reg = sm.OLS(y, X).fit()

# FF
y, X = dmatrices('Equal_weight_returns_minus_RF ~ Mkt_RF + SMB + HML', data=ff_market_reg, return_type='dataframe')
ff_reg = sm.OLS(y, X).fit()

final_output.loc['Capm Alpha','Equal Weighted'] =  capm_reg.params[0]
final_output.loc['Capm Alpha t-stat','Equal Weighted'] = capm_reg.tvalues.Intercept
final_output.loc['Sharpe Ratio','Equal Weighted'] =  y.mean()[0]/y.std()[0] * math.sqrt(252)
final_output.loc['3 Factor FF Alpha','Equal Weighted'] =  ff_reg.params[0]
final_output.loc['3 Factor Alpha t-stat','Equal Weighted'] = ff_reg.tvalues.Intercept


## Value Weights

finding_mcap = daily_returns[['PERMNO', 'date', 'PRC', 'RET', 'SHROUT', 'CFACPR', 'CFACSHR']]
finding_mcap.sort_values(['PERMNO','date'],inplace=True)
finding_mcap['date'] = finding_mcap.groupby(['PERMNO'])['date'].shift(-1)
finding_mcap.dropna(inplace = True)
finding_mcap['mcap'] = finding_mcap['PRC']* finding_mcap['SHROUT']
finding_mcap= finding_mcap[['PERMNO', 'date', 'mcap']]
finding_mcap = finding_mcap[finding_mcap.PERMNO.isin(stock_splits.PERMNO)]
finding_mcap = pd.pivot_table(finding_mcap, values = 'mcap', columns = 'PERMNO', index = 'date').fillna(0.0)
finding_mcap = finding_mcap.mask(finding_mcap < 0, 0)
finding_mcap_v2 = finding_mcap.mask(stock_split_returns_v2 == 0, 0)

# finding weights

value_weights_daywise = finding_mcap_v2.copy()
value_weights_daywise['Total_Mcap'] = finding_mcap_v2.sum(axis=1)

for i in finding_mcap_v2.columns.values:
    value_weights_daywise[i] = value_weights_daywise[i]/value_weights_daywise['Total_Mcap']

value_weights_daywise = value_weights_daywise[finding_mcap.columns.values]


# VW Portfolio returns
vw_returns = value_weights_daywise*stock_split_returns_v2
portfolio_return['Value_weight_returns'] = vw_returns.sum(axis=1)


ff_market_reg = pd.merge(left = ff_returns, right = portfolio_return.reset_index(), left_on = ['Date'], right_on = ['date'] ).fillna(1.0)[['Date','Mkt_RF','Value_weight_returns','RF','SMB','HML']]
ff_market_reg['Value_weight_returns_minus_RF'] = ff_market_reg['Value_weight_returns'] - ff_market_reg['RF']

y, X = dmatrices('Value_weight_returns_minus_RF ~ Mkt_RF', data=ff_market_reg, return_type='dataframe')
capm_reg = sm.OLS(y, X).fit()

y, X = dmatrices('Value_weight_returns_minus_RF ~ Mkt_RF + SMB + HML', data=ff_market_reg, return_type='dataframe')
ff_reg = sm.OLS(y, X).fit()

final_output.loc['Capm Alpha','Value Weighted'] =  capm_reg.params[0]
final_output.loc['Capm Alpha t-stat','Value Weighted'] = capm_reg.tvalues.Intercept
final_output.loc['Sharpe Ratio','Value Weighted'] =  y.mean()[0]/y.std()[0] * math.sqrt(252)
final_output.loc['3 Factor FF Alpha','Value Weighted'] =  ff_reg.params[0] 
final_output.loc['3 Factor Alpha t-stat','Value Weighted'] = ff_reg.tvalues.Intercept



#### Part 3
### Market Returns Removing Top 500 stocks by Market Cap (Daily Basis)

daily_returns_exc_500 =  daily_returns.copy()
daily_returns_exc_500.RET = daily_returns_exc_500.RET.astype(float)
daily_returns_exc_500['mcap'] = daily_returns_exc_500.SHROUT * daily_returns_exc_500.PRC
daily_returns_exc_500['rank'] = daily_returns_exc_500.groupby('date')['mcap'].rank('dense', ascending = False)
daily_returns_exc_500 = daily_returns_exc_500[daily_returns_exc_500['rank'] > 500]

# Equal Weight Market Return
ff_returns['Mkt_exc500_ew'] = daily_returns_exc_500.groupby('date')['RET'].mean()
ff_returns['Mkt_exc500_ew_RF'] = ff_returns['Mkt_exc500_ew'] - ff_returns['RF']

ff_market_reg = pd.merge(left = ff_returns, right = portfolio_return.reset_index(), left_on = ['Date'], right_on = ['date'] ).fillna(1.0)[['Date','Mkt_exc500_ew_RF','Equal_weight_returns','RF','SMB','HML']]
ff_market_reg['Equal_weight_returns_minus_RF'] = ff_market_reg['Equal_weight_returns'] - ff_market_reg['RF']

y, X = dmatrices('Equal_weight_returns_minus_RF ~ Mkt_exc500_ew_RF', data=ff_market_reg, return_type='dataframe')
capm_reg = sm.OLS(y, X).fit()

y, X = dmatrices('Equal_weight_returns_minus_RF ~ Mkt_exc500_ew_RF + SMB + HML', data=ff_market_reg, return_type='dataframe')
ff_reg = sm.OLS(y, X).fit()

final_output_exc500.loc['Capm Alpha','Equal Weighted'] =  capm_reg.params[0]
final_output_exc500.loc['Capm Alpha t-stat','Equal Weighted'] = capm_reg.tvalues.Intercept
final_output_exc500.loc['Sharpe Ratio','Equal Weighted'] =  y.mean()[0]/y.std()[0] * math.sqrt(252)
final_output_exc500.loc['3 Factor FF Alpha','Equal Weighted'] =  ff_reg.params[0] 
final_output_exc500.loc['3 Factor Alpha t-stat','Equal Weighted'] = ff_reg.tvalues.Intercept


# Value Weight Market Return (Change in total Market Cap)
daily_total_mcap = daily_returns_exc_500.groupby('date')['mcap'].sum()
ff_returns['Mkt_exc500_vw'] = daily_total_mcap.pct_change().fillna(1.0)
ff_returns['Mkt_exc500_vw_RF'] = ff_returns['Mkt_exc500_vw'] - ff_returns['RF']

ff_market_reg = pd.merge(left = ff_returns, right = portfolio_return.reset_index(), left_on = ['Date'], right_on = ['date'] ).fillna(1.0)[['Date','Mkt_exc500_vw_RF','Value_weight_returns','RF','SMB','HML']]
ff_market_reg['Value_weight_returns_minus_RF'] = ff_market_reg['Value_weight_returns'] - ff_market_reg['RF']

y, X = dmatrices('Value_weight_returns_minus_RF ~ Mkt_exc500_vw_RF', data=ff_market_reg, return_type='dataframe')
capm_reg = sm.OLS(y, X).fit()

y, X = dmatrices('Value_weight_returns_minus_RF ~ Mkt_exc500_vw_RF + SMB + HML', data=ff_market_reg, return_type='dataframe')
ff_reg = sm.OLS(y, X).fit()

final_output_exc500.loc['Capm Alpha','Value Weighted'] =  capm_reg.params[0]
final_output_exc500.loc['Capm Alpha t-stat','Value Weighted'] = capm_reg.tvalues.Intercept
final_output_exc500.loc['Sharpe Ratio','Value Weighted'] =  y.mean()[0]/y.std()[0] * math.sqrt(252)
final_output_exc500.loc['3 Factor FF Alpha','Value Weighted'] =  ff_reg.params[0] 
final_output_exc500.loc['3 Factor Alpha t-stat','Value Weighted'] = ff_reg.tvalues.Intercept