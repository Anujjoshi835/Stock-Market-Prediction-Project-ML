#**************** IMPORT PACKAGES ********************
from flask import Flask, render_template, request
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math
from datetime import datetime
import yfinance as yf
from sklearn.linear_model import LinearRegression
import constants as ct
import nltk
import reddit_sentiment
nltk.download('punkt')
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#***************** FLASK *****************************
app = Flask(__name__)

@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return render_template('index.html')

def get_historical(quote):
    end = datetime.now()
    start = datetime(end.year-2, end.month, end.day)
    data = yf.download(quote, start=start, end=end)
    df = pd.DataFrame(data=data)
    df.to_csv(f'{quote}.csv')
    if df.empty:
        ts = TimeSeries(key='N6A6QT6IBFJOPJ70', output_format='pandas')
        data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
        data = data.head(503).iloc[::-1]
        data = data.reset_index()
        df = pd.DataFrame()
        df['Date'] = data['date']
        df['Open'] = data['1. open']
        df['High'] = data['2. high']
        df['Low'] = data['3. low']
        df['Close'] = data['4. close']
        df['Adj Close'] = data['5. adjusted close']
        df['Volume'] = data['6. volume']
        df.to_csv(f'{quote}.csv', index=False)
    return

def ARIMA_ALGO(df, quote):
    uniqueVals = df["Code"].unique()
    df = df.set_index("Code")
    def parser(x):
        return datetime.strptime(x, '%Y-%m-%d')
    def arima_model(train, test):
        history = [x for x in train]
        predictions = []
        for t in range(len(test)):
            model = ARIMA(history, order=(6,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
        return predictions
    company = uniqueVals[0]
    data = (df.loc[company, :]).reset_index()
    data['Price'] = data['Close']
    # Fix for missing 'Date' column
    if 'Date' not in data.columns and 'index' in data.columns:
        data = data.rename(columns={'index': 'Date'})
    Quantity_date = data[['Price', 'Date']]
    Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
    Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
    Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
    Quantity_date = Quantity_date.drop(['Date'], axis=1)
    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(Quantity_date)
    plt.savefig('static/Trends.png')
    plt.close(fig)
    quantity = Quantity_date.values
    size = int(len(quantity) * 0.80)
    train, test = quantity[0:size], quantity[size:len(quantity)]
    predictions = arima_model(train, test)
    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(test, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.legend(loc=4)
    plt.savefig('static/ARIMA.png')
    plt.close(fig)
    arima_pred = predictions[-2]
    error_arima = math.sqrt(mean_squared_error(test, predictions))
    return arima_pred, error_arima

def LSTM_ALGO(df, quote):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM
    from sklearn.preprocessing import MinMaxScaler

    dataset_train = df.iloc[0:int(0.8*len(df)), :]
    dataset_test = df.iloc[int(0.8*len(df)):, :]
    training_set = df.iloc[:, 4:5].values
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)
    X_train, y_train = [], []
    for i in range(7, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-7:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_forecast = np.array(X_train[-1, 1:])
    X_forecast = np.append(X_forecast, y_train[-1])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))
    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.1))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
    real_stock_price = dataset_test.iloc[:, 4:5].values
    dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
    testing_set = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values
    testing_set = testing_set.reshape(-1, 1)
    testing_set = sc.transform(testing_set)
    X_test = []
    for i in range(7, len(testing_set)):
        X_test.append(testing_set[i-7:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(real_stock_price, label='Actual Price')
    plt.plot(predicted_stock_price, label='Predicted Price')
    plt.legend(loc=4)
    plt.savefig('static/LSTM.png')
    plt.close(fig)
    error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    forecasted_stock_price = regressor.predict(X_forecast)
    forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)
    lstm_pred = forecasted_stock_price[0, 0]
    return lstm_pred, error_lstm

def LIN_REG_ALGO(df, quote):
    from sklearn.preprocessing import StandardScaler
    forecast_out = int(7)
    df['Close after n days'] = df['Close'].shift(-forecast_out)
    df_new = df[['Close', 'Close after n days']]
    y = np.array(df_new.iloc[:-forecast_out, -1])
    y = np.reshape(y, (-1, 1))
    X = np.array(df_new.iloc[:-forecast_out, 0:-1])
    X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:, 0:-1])
    X_train = X[0:int(0.8*len(df)), :]
    X_test = X[int(0.8*len(df)):, :]
    y_train = y[0:int(0.8*len(df)), :]
    y_test = y[int(0.8*len(df)):, :]
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_to_be_forecasted = sc.transform(X_to_be_forecasted)
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    y_test_pred = y_test_pred * (1.04)
    import matplotlib.pyplot as plt2
    fig = plt2.figure(figsize=(7.2, 4.8), dpi=65)
    plt2.plot(y_test, label='Actual Price')
    plt2.plot(y_test_pred, label='Predicted Price')
    plt2.legend(loc=4)
    plt2.savefig('static/LR.png')
    plt2.close(fig)
    error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
    forecast_set = clf.predict(X_to_be_forecasted)
    forecast_set = forecast_set * (1.04)
    mean = forecast_set.mean()
    lr_pred = forecast_set[0, 0]
    return df, lr_pred, forecast_set, mean, error_lr

def retrieving_reddit_polarity(symbol):
    stock_ticker_map = pd.read_csv('Yahoo.csv')
    stock_full_form = stock_ticker_map[stock_ticker_map['Ticker'] == symbol]
    if stock_full_form.empty:
        return 0, [], "No company found for symbol", 0, 0, 0
    company_name = stock_full_form['Name'].to_list()[0][:12]
    result = reddit_sentiment.get_reddit_sentiment(company_name, limit=ct.num_of_tweets)
    polarity = result['average_polarity']
    tw_list = result['texts']
    pos = result['positive_count']
    neg = result['negative_count']
    neutral = result['neutral_count']
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [pos, neg, neutral]
    explode = (0, 0, 0)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=65)
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig('static/SA.png')
    plt.close(fig)
    if polarity > 0:
        tw_pol = "Overall Positive"
    elif polarity < 0:
        tw_pol = "Overall Negative"
    else:
        tw_pol = "Neutral"
    return polarity, tw_list, tw_pol, pos, neg, neutral

def recommending(df, global_polarity, today_stock, mean, quote):
    if today_stock.iloc[-1]['Close'] < mean:
        if global_polarity > 0:
            idea = "RISE"
            decision = "BUY"
        else:
            idea = "FALL"
            decision = "SELL"
    else:
        idea = "FALL"
        decision = "SELL"
    return idea, decision

@app.route('/insertintotable', methods=['POST'])
def insertintotable():
    nm = request.form['nm']
    quote = nm
    try:
        get_historical(quote)
    except Exception:
        return render_template('index.html', not_found=True)
    else:
        df = pd.read_csv(f'{quote}.csv')
        today_stock = df.iloc[-1:]
        df = df.dropna()
        code_list = [quote] * len(df)
        df2 = pd.DataFrame(code_list, columns=['Code'])
        df2 = pd.concat([df2, df], axis=1)
        df = df2
        arima_pred, error_arima = ARIMA_ALGO(df, quote)
        lstm_pred, error_lstm = LSTM_ALGO(df, quote)
        df, lr_pred, forecast_set, mean, error_lr = LIN_REG_ALGO(df, quote)
        polarity, tw_list, tw_pol, pos, neg, neutral = retrieving_reddit_polarity(quote)
        idea, decision = recommending(df, polarity, today_stock, mean, quote)
        today_stock = today_stock.round(2)
        return render_template(
            'results.html',
            quote=quote,
            arima_pred=round(arima_pred, 2),
            lstm_pred=round(lstm_pred, 2),
            lr_pred=round(lr_pred, 2),
            open_s=today_stock['Open'].to_string(index=False),
            close_s=today_stock['Close'].to_string(index=False),
            adj_close=today_stock['Adj Close'].to_string(index=False),
            tw_list=tw_list,
            tw_pol=tw_pol,
            idea=idea,
            decision=decision,
            high_s=today_stock['High'].to_string(index=False),
            low_s=today_stock['Low'].to_string(index=False),
            vol=today_stock['Volume'].to_string(index=False),
            forecast_set=forecast_set,
            error_lr=round(error_lr, 2),
            error_lstm=round(error_lstm, 2),
            error_arima=round(error_arima, 2)
        )

if __name__ == '__main__':
    app.run()

















