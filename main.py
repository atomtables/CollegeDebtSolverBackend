import csv
import datetime
import numpy as np
import pandas as pd
from polygon import RESTClient
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import tensorflow as tf


ticker1 = ''
polygon_api_key = "YzQk9Qzpr5jIGOUhPH3p5YQywSjizNEp"
client = RESTClient(polygon_api_key)
nasdaq_tickers = pd.read_csv('nasdaq_screener_1717938946132.csv')
nasdaq_tickers1 = nasdaq_tickers.drop(columns=['Name','Last Sale','Net Change','% Change','Market Cap','Country','IPO Year','Volume','Sector','Industry'])
stockwanted = input("Enter the stock symbol: ")
stockTicker = []


for i ,row in nasdaq_tickers1.iterrows():
    stockTicker.append(row[0])


while True:
    if stockwanted in stockTicker:
        ticker1 = stockwanted
        break 
    else:
        print("Stock not found")
      
      
    

dataRequest = client.get_aggs(ticker=ticker1,multiplier=30,timespan='day',from_='2022-09-01', to='2100-01-01')
#print(dataRequest)

fields=["open","high","low","close","volume","vwap","timestamp","transactions","otc"]

priceData = pd.DataFrame(dataRequest)
priceData['Date'] = priceData['timestamp'].apply(
    lambda x: pd.to_datetime(x * 1000000))


X = priceData.drop(columns=['open','high','low','close','volume','vwap','transactions','otc', 'timestamp'])


y= priceData.drop(columns=['open','high','low','volume','vwap','transactions','otc', 'timestamp','Date'])



train_test_split(X,y,test_size=0.2)




df=priceData[["Date","close"]]
df.index=df.pop("Date")

print(df)



def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime.datetime(year=year, month=month, day=day)





def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
  first_date = str_to_datetime(first_date_str)
  last_date  = str_to_datetime(last_date_str)

  target_date = first_date

  dates = []
  X, Y = [], []

  last_time = False
  while True:
    df_subset = dataframe.loc[:target_date].tail(n+1)

    if len(df_subset) != n+1:
      print(f'Error: Window of size {n} is too large for date {target_date}')
      return

    values = df_subset['Close'].to_numpy()
    x, y = values[:-1], values[-1]

    dates.append(target_date)
    X.append(x)
    Y.append(y)

    next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
    next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
    next_date_str = next_datetime_str.split('T')[0]
    year_month_day = next_date_str.split('-')
    year, month, day = year_month_day
    next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

    if last_time:
      break

    target_date = next_date

    if target_date == last_date:
      last_time = True

  ret_df = pd.DataFrame({})
  ret_df['Target Date'] = dates

  X = np.array(X)
  for i in range(0, n):
    X[:, i]
    ret_df[f'Target-{n-i}'] = X[:, i]

  ret_df['Target'] = Y

  return ret_df

# Start day second time around: '2021-03-25'
windowed_df = df_to_windowed_df(df, 
                                "2022-09-01-04", 
                                '2022-10-01-04', 
                                n=3)
    

def windowed_df_to_date_X_y(windowed_dataframe):
  df_as_np = windowed_dataframe.to_numpy()

  dates = df_as_np[:, 0]

  middle_matrix = df_as_np[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)

dates, X, y = windowed_df_to_date_X_y(windowed_df)


q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)

dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

plt.plot(dates_train, y_train)
plt.plot(dates_val, y_val)
plt.plot(dates_test, y_test)
plt.legend(['Train', 'Validation', 'Test'])
plt.show()

from tensorflow import keras
model = keras.models.Sequential([
    keras.layers.Input((3, 1)),
    keras.layers.LSTM(64),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])


from tensorflow.keras.optimizers import Adam
model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)









