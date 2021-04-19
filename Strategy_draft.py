from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from math import log, isnan
from statistics import stdev
from scipy.stats.mstats import gmean


N = 10
alpha = .02
threshold = .001
lot_size = 100


US_data = pd.read_csv('US_data.csv')
US_data['Date'] = pd.to_datetime(US_data['Date'])
HK_data = pd.read_csv('HK_data.csv')
HK_data['Date'] = pd.to_datetime(HK_data['Date'])



#linear regression and GMRR
##add R^2 back in
HK_features = []
for dt in HK_data['Date'][N:]:
    eod_close_prices = list(HK_data['Last Price'][HK_data['Date'] <= dt].tail(N))
    action = []
    index = np.array(list(i for i in range(1, len(eod_close_prices))))
    log_returns = np.array(list(log(i / j) for i, j in zip(eod_close_prices[:N - 1], eod_close_prices[1:])))
    ret = []
    for i in log_returns[:-1]:
        ret.append(i+1)
    GMRR = np.prod(ret)** (1 / len(log_returns)) - 1
    if log_returns[-1] - GMRR > threshold:
        action.append('BUY')
    else:
        action.append('SELL')
    index = index.reshape(1,-1)
    log_returns = log_returns.reshape(1,-1)
    linreg_model = linear_model.LinearRegression()
    linreg_model.fit(index, log_returns)
    modeled_returns = linreg_model.predict(log_returns)
    row = [dt, linreg_model.intercept_, linreg_model.coef_[0][0],GMRR, action]
    HK_features.append(row)

      #r2_score(log_returns[:N],modeled_returns),
HK_features = pd.DataFrame(HK_features)
HK_features.columns = ["Date", "a", "b", "GMRR", "Action"]
HK_features['Date'] = pd.to_datetime(HK_features['Date'])


# Get available volatility of day-over-day log returns based on closing
#     # prices for IVV using a window size of N days.
ivv_features = []
for dt in US_data['Date'][N:]:
    eod_close_prices = list(US_data['Last Price'][US_data['Date'] <= dt].tail(N))
    vol = stdev([
             log(i / j) for i, j in zip(
                 eod_close_prices[:N - 1], eod_close_prices[1:]
             )
         ])
    vol_row = [dt, vol]
    ivv_features.append(vol_row)

ivv_features = pd.DataFrame(ivv_features)
ivv_features.columns = ["Date", "ivv_vol"]
ivv_features['Date'] = pd.to_datetime(ivv_features['Date'])

#     #inner merge on features from IVV and the bond rates,
features = pd.merge(HK_features, ivv_features, on='Date')
#
#     # delete vars we no longer need
del HK_data
del HK_features
del ivv_features
#
response = []



#
# def backtest(US_data, HK_data, N, alpha, threshold, lot_size, start_date, end_date, starting_cash):
#
#     # linear regression on log returns
#     def HK_linreg(HK_data):
#         log_returns = []
#         index = []
#         for i in HK_data['Date'][N:]:
#             eod_close_prices = list(HK_data['Low Price'][HK_data['Date'] <= dt].tail(N))
#             log_returns[i] = log(eod_close_prices(i+1)/eod_close_prices(i))
#             #log_returns = (log(i / j) for i, j in zip(eod_close_prices[:N - 1], eod_close_prices[1:]))
#             index[i] = i
#         linreg_model = linear_model.LinearRegression()
#         linreg_model.fit(index, log_returns)
#         modeled_returns = linreg_model.predict(log_returns)
#         return [HK_data.Date, linreg_model.coef_[0],
#                 linreg_model.intercept_,
#                 r2_score(log_returns[1:],
#                          modeled_returns)]
#
#     # apply bonds_fun to every row in bonds_hist to make the features dataframe.
#     HK_features = HK_linreg(HK_data)
#     HK_features.columns = ["Date", "a", "b", "R2"]
#     HK_features.Date = pd.to_datetime(HK_features.Date)
#
#     # Get available volatility of day-over-day log returns based on closing
#     # prices for IVV using a window size of N days.
#     ivv_features = []
#
#     for dt in US_data['Date'][N:]:
#         eod_close_prices = list(
#             US_data['Low Price'][US_data['Date'] <= dt].tail(N))
#         vol = stdev([
#             log(i / j) for i, j in zip(
#                 eod_close_prices[:N - 1], eod_close_prices[1:]
#             )
#         ])
#         vol_row = [dt, vol]
#         ivv_features.append(vol_row)
#
#     ivv_features = pd.DataFrame(ivv_features)
#     ivv_features.columns = ["Date", "ivv_vol"]
#     ivv_features['Date'] = pd.to_datetime(ivv_features['Date'])
#
#     #inner merge on features from IVV and the bond rates,
#     features = pd.merge(HK_features, ivv_features, on='Date')
#
#     # delete vars we no longer need
#     del HK_data
#     del HK_features
#     del ivv_features
#
#    response = []