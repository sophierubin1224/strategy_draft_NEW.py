from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from math import log, isnan
from statistics import stdev
from numpy import repeat

def trading_decision(
        exit_date, response_var, features_and_responses, trading_date, N, n
):
    training_indices = features_and_responses[exit_date] < trading_date
    training_X = features_and_responses[training_indices].tail(N)[
        ['a', 'b', 'R2', 'ivv_vol']
    ]
    training_Y = features_and_responses[training_indices].tail(N)[response_var]

    # Need at least two 1's to train a model
    if sum(training_Y) < 2:
        return 0

    if sum(training_Y) < n:
        logisticRegr = linear_model.LogisticRegression()
        logisticRegr.fit(np.float64(training_X), np.float64(training_Y))
        trade_decision = logisticRegr.predict(
            np.float64(
                features_and_responses[["a", "b", "R2", "ivv_vol"]][
                    features_and_responses['Date'] == trading_date
                    ]
            )
        ).item()
    else:     # If EVERYTHING is a 1, then just go ahead and implement again.
        trade_decision = 1

    return trade_decision

def backtest(
        US_data, HK_data, n, N, alpha, lot_size, start_date, end_date,
        starting_cash
):

    print(US_data)
    print(HK_data)

    # Convert JSON data to dataframes
    US_data = pd.read_json(US_data)
    HK_data = pd.read_json(HK_data)
    # Create the features data frame from the bond yields & IVV hist data

    # linear regression on log returns
    def HK_linreg(HK_data):
        log_returns = []
        index = []
        for i in HK_data:
            log_returns[i] = np.log(HK_data[i+1].VWAP/HK_data[i].VWAP)
            index[i] = i
        linreg_model = linear_model.LinearRegression()
        linreg_model.fit(index, log_returns)
        modeled_returns = linreg_model.predict(log_returns)
        return [HK_data.Date, linreg_model.coef_[0],
                linreg_model.intercept_,
                r2_score(log_returns[1:],
                        modeled_returns)]

    # apply bonds_fun to every row in bonds_hist to make the features dataframe.
    HK_features = HK_linreg(HK_data)
    HK_features.columns = ["Date", "a", "b", "R2"]
    HK_features.Date = pd.to_datetime(HK_features.Date)

    # Get available volatility of day-over-day log returns based on closing
    # prices for IVV using a window size of N days.
    ivv_features = []

    for dt in US_data['Date'][N:]:
        eod_close_prices = list(
            US_data['Close'][US_data['Date'] <= dt].tail(N))
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

    # here, I'm doing an inner merge on features from IVV and the bond rates,
    # storing the result in a dataframe called 'features'.
    # The reason is because federal and NYSE holidays are not exactly the same, so
    # there are some days on which the federal government reports bond features
    # but no IVV data exists, and vice versa.
    features = pd.merge(HK_features, ivv_features, on='Date')

    # delete vars we no longer need
    del HK_data
    del HK_features
    del ivv_features

    response = []

    for features_dt in features['Date']:
        # Get data for the next n days after response_date
        ohlc_data = US_data[['Date', 'Open', 'High', 'Low', 'Close']][
            US_data['Date'] > features_dt
            ].head(n)

        if len(ohlc_data) == 0:
            response_row = repeat(None, 8).tolist()
            response.append(response_row)
            continue

        less_than_n_ohlc_data_rows = len(ohlc_data) < n

        entry_date = ohlc_data['Date'].head(1).item()
        entry_price = ohlc_data['Open'].head(1).item()

        target_price_long = entry_price * (1 + alpha)
        target_price_short = entry_price * (1 - alpha)

        high_price = max(ohlc_data['High'])
        low_price = min(ohlc_data['Low'])

        long_success = int(high_price >= target_price_long)
        short_success = int(low_price <= target_price_short)

        exit_long = next(
            (y.values.tolist() for x, y in
             ohlc_data[['Date', 'High']].iterrows()
             if y[1] >= target_price_long),
            ohlc_data[['Date', 'High']].tail(1).values.tolist()[0]
        )

        exit_short = next(
            (y.values.tolist() for x, y in ohlc_data[['Date', 'Low']].iterrows()
             if y[1] <= target_price_short),
            ohlc_data[['Date', 'Low']].tail(1).values.tolist()[0]
        )

        response_row = [entry_date, entry_price, long_success, short_success] + \
                       exit_long + exit_short

        if less_than_n_ohlc_data_rows:
            if not bool(response_row[2]):
                for i in [2, 4, 6]:
                    response_row[i] = None
            if not bool(response_row[3]):
                for i in [3, 5, 7]:
                    response_row[i] = None

        response.append(response_row)

    response = pd.DataFrame(response)
    response.columns = ["entry_date", "entry_price", "long_success",
                        "short_success", "exit_date_long", "exit_price_long",
                        "exit_date_short", "exit_price_short"]
    response = response.round(2)

    features_and_responses = pd.concat([features, response], axis=1)
    del features
    del response

    blotter = []
    trade_id = 0

    for trading_date in features_and_responses['Date'][
        features_and_responses['Date'] >= pd.to_datetime(start_date)
    ]:
        trade_decision_long = trading_decision(
            'exit_date_long', 'long_success', features_and_responses,
            trading_date, N, n
        )
        # trade_decision_short = trading_decision(
        #     'exit_date_short', 'short_success', features_and_responses,
        #     trading_date, N, n
        # )

        # trade_sum = trade_decision_short + trade_decision_long

        # if trade_sum == 0 or trade_sum == 2:
        #     continue

        if trade_decision_long == 1:
            right_answer = features_and_responses[
                features_and_responses['Date'] == trading_date
                ]

            if trading_date == features_and_responses['Date'].tail(1).item():
                order_status = 'PENDING'
                submitted = order_price = fill_price = filled_or_cancelled = None
            else:
                submitted = filled_or_cancelled = right_answer[
                    'entry_date'].item()
                order_price = fill_price = right_answer['entry_price'].item()
                order_status = 'FILLED'

            entry_trade_mkt = [
                trade_id, 'L', submitted, 'BUY', lot_size, 'IVV',
                order_price, 'MKT', order_status, fill_price,
                filled_or_cancelled
            ]

            long_success = right_answer['long_success'].item()

            if isnan(long_success):
                order_status = 'OPEN'
                fill_price = filled_or_cancelled = None

            filled_or_cancelled = right_answer['exit_date_long'].item()

            if isinstance(order_price, float):
                order_price = order_price * (1+alpha)

            if long_success == 0:
                order_status = 'CANCELLED'
                fill_price = None
                exit_trade_mkt = [
                    trade_id, 'L', filled_or_cancelled, 'SELL', lot_size,
                    'IVV', right_answer['exit_price_long'].item(), 'MKT',
                    'FILLED', right_answer['exit_price_long'].item(),
                    filled_or_cancelled
                ]
                blotter.append(exit_trade_mkt)

            if long_success == 1:
                order_status = 'FILLED'
                fill_price = right_answer['exit_price_long'].item()

            exit_trade_lmt = [
                trade_id, 'L', submitted, 'SELL', lot_size, 'IVV',
                order_price, 'LIMIT', order_status, fill_price,
                filled_or_cancelled
            ]

            blotter.append(entry_trade_mkt)
            blotter.append(exit_trade_lmt)
            trade_id += 1

        # elif trade_decision_short == 1:
        #     right_answer = features_and_responses[
        #         features_and_responses['Date'] == trading_date
        #         ]
        #
        #     if trading_date == features_and_responses['Date'].tail(1).item():
        #         order_status = 'PENDING'
        #         submitted = order_price = fill_price = filled_or_cancelled = None
        #     else:
        #         submitted = filled_or_cancelled = right_answer[
        #             'entry_date'].item()
        #         order_price = fill_price = right_answer['entry_price'].item()
        #         order_status = 'FILLED'
        #
        #     entry_trade_mkt = [
        #         trade_id, 'S', submitted, 'SELL', lot_size, 'IVV',
        #         order_price, 'MKT', order_status, fill_price,
        #         filled_or_cancelled
        #     ]
        #
        #     short_success = right_answer['short_success'].item()
        #
        #     if isnan(short_success):
        #         order_status = 'OPEN'
        #         fill_price = filled_or_cancelled = None
        #
        #     filled_or_cancelled = right_answer['exit_date_short'].item()
        #
        #     if isinstance(order_price, float):
        #         order_price = order_price * (1-alpha)
        #
        #     if short_success == 0:
        #         order_status = 'CANCELLED'
        #         fill_price = None
        #         exit_trade_mkt = [
        #             trade_id, 'S', filled_or_cancelled, 'BUY', lot_size,
        #             'IVV', right_answer['exit_price_short'].item(), 'MKT',
        #             'FILLED', right_answer['exit_price_short'].item(),
        #             filled_or_cancelled
        #         ]
        #         blotter.append(exit_trade_mkt)
        #
        #     if short_success == 1:
        #         order_status = 'FILLED'
        #         fill_price = right_answer['exit_price_short'].item()
        #
        #     exit_trade_lmt = [
        #         trade_id, 'S', submitted, 'BUY', lot_size, 'IVV',
        #         order_price, 'LIMIT', order_status, fill_price,
        #         filled_or_cancelled
        #     ]
        #
        #     blotter.append(entry_trade_mkt)
        #     blotter.append(exit_trade_lmt)
        #     trade_id += 1

    blotter = pd.DataFrame(blotter)
    blotter.columns = [
        'ID', 'ls', 'submitted', 'action', 'size', 'symbol', 'price', 'type',
        'status', 'fill_price', 'filled_or_cancelled'
    ]
    blotter = blotter.round(2)
    blotter.sort_values(
        by=['ID', 'submitted'],
        inplace=True,
        ascending=[False, True]
    )
    blotter.reset_index()

    calendar_ledger = []
    cash = starting_cash
    position = 0
    stock_value = 0
    total_value = cash

    for ivv_row in US_data[
        US_data['Date'] >= pd.to_datetime(start_date)
    ].iterrows():
        trading_date = ivv_row[1]['Date']
        ivv_close = ivv_row[1]['Close']
        trades = blotter[
            (blotter['filled_or_cancelled'] == ivv_row[1]['Date']) & (
                    blotter['status'] == 'FILLED'
            )]
        if len(trades) > 0:
            position = position + sum(
                trades['size'][trades['action'] == 'BUY']
            ) - sum(
                trades['size'][trades['action'] == 'SELL']
            )
            cash = cash - sum(
                trades['size'][trades['action'] == 'BUY'] *
                trades['fill_price'][
                    trades['action'] == 'BUY'
                    ]
            ) + sum(
                trades['size'][trades['action'] == 'SELL'] *
                trades['fill_price'][
                    trades['action'] == 'SELL'
                    ]
            )
            stock_value = position * ivv_close
            total_value = cash + stock_value
        else:
            stock_value = position * ivv_close
            total_value = cash + stock_value

        ledger_row = [
            trading_date, position, ivv_close, cash, stock_value, total_value
        ]
        calendar_ledger.append(ledger_row)

    calendar_ledger = pd.DataFrame(calendar_ledger)
    calendar_ledger.columns = [
        'Date', 'position', 'ivv_close', 'cash', 'stock_value', 'total_value'
    ]

    trade_ledger = []

    for trade in blotter['ID'].unique():
        round_trip_trade = blotter[
            (blotter['ID'] == trade) & (blotter['status'] == 'FILLED')
            ]

        if len(round_trip_trade) < 2:
            continue

        trade_id = round_trip_trade['ID'].unique().item()

        date_opened = min(round_trip_trade['submitted'])
        date_closed = max(round_trip_trade['submitted'])

        ivv_df = US_data[(US_data['Date'] <= date_closed) & (
                US_data['Date'] >= date_opened
            )]

        trading_days_open = len(ivv_df)

        buy_price = round_trip_trade['fill_price'][
            round_trip_trade['action'] == 'BUY'
            ].item()
        sell_price = round_trip_trade['fill_price'][
            round_trip_trade['action'] == 'SELL'
            ].item()

        ivv_price_enter = ivv_df['Close'][
            ivv_df['Date'] == round_trip_trade['submitted'][
                round_trip_trade['action'] == 'BUY'
            ].item()
        ].item()
        ivv_price_exit = ivv_df['Close'][
            ivv_df['Date'] == round_trip_trade['submitted'][
                round_trip_trade['action'] == 'SELL'
            ].item()
        ].item()

        trade_rtn = log(sell_price / buy_price)
        ivv_rtn = log(ivv_price_exit / ivv_price_enter)

        trade_rtn_per_trading_day = trade_rtn/trading_days_open
        benchmark_rtn_per_trading_day   = ivv_rtn/trading_days_open

        trade_ledger_row = [
            trade_id, date_opened, date_closed, trading_days_open, buy_price,
            sell_price, ivv_price_enter, ivv_price_exit, trade_rtn, ivv_rtn,
            trade_rtn_per_trading_day, benchmark_rtn_per_trading_day
        ]

        trade_ledger.append(trade_ledger_row)

    trade_ledger = pd.DataFrame(trade_ledger)
    trade_ledger.columns = [
        'trade_id', 'open_dt', 'close_dt', 'trading_days_open', 'buy_price',
        'sell_price', 'benchmark_buy_price', 'benchmark_sell_price',
        'trade_rtn', 'benchmark_rtn', 'trade_rtn_per_trading_day',
        'benchmark_rtn_per_trading_day'
    ]

    # Final formatting
    features_and_responses['Date'] = features_and_responses['Date'].dt.date
    features_and_responses['entry_date'] = features_and_responses[
        'entry_date'].dt.date
    features_and_responses['exit_date_long'] = features_and_responses[
        'exit_date_long'].dt.date
    features_and_responses['exit_date_short'] = features_and_responses[
        'exit_date_short'].dt.date

    blotter['submitted'] = blotter['submitted'].dt.date
    blotter['filled_or_cancelled'] = blotter['filled_or_cancelled'].dt.date

    calendar_ledger['Date'] = calendar_ledger['Date'].dt.date
    calendar_ledger.round(2)

    trade_ledger['open_dt'] = trade_ledger['open_dt'].dt.date
    trade_ledger['close_dt'] = trade_ledger['close_dt'].dt.date

    features_and_responses.to_csv('features_and_responses.csv')
    blotter.to_csv('blotter.csv')
    calendar_ledger.to_csv('calendar_ledger.csv')
    trade_ledger.to_csv('trade_ledger.csv')

    return features_and_responses, blotter, calendar_ledger, trade_ledger
