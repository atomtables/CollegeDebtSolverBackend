import yfinance as yf


def make_index_timezone_naive(df):
  """
  Convert the index of a DataFrame from timezone-aware to timezone-naive datetime objects.

  Parameters:
  df (pd.DataFrame): DataFrame with a timezone-aware datetime index.

  Returns:
  pd.DataFrame: DataFrame with a timezone-naive datetime index.
  """
  if df.index.tz is not None:
    df.index = df.index.tz_localize(None)
  return df


stocks = yf
msft = yf.Ticker("MSFT")

# get historical data
hist = msft.history(period="5y")
hist = make_index_timezone_naive(hist)
print(hist)

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import datetime
import pytz


def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  dt = datetime.datetime(year=year, month=month, day=day)
  return dt


def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
  first_date = str_to_datetime(first_date_str)
  last_date = str_to_datetime(last_date_str)

  target_date = first_date

  dates = []
  X, Y = [], []

  last_time = False
  while True:
    df_subset = dataframe.loc[:target_date].tail(n + 1)

    if len(df_subset) != n + 1:
      print(f'Error: Window of size {n} is too large for date {target_date}')
      continue

    values = df_subset['Close'].to_numpy()
    x, y = values[:-1], values[-1]

    dates.append(target_date)
    X.append(x)
    Y.append(y)

    next_week = dataframe.loc[target_date:target_date +
                              datetime.timedelta(days=7)]
    next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
    next_date_str = next_datetime_str.split('T')[0]
    year_month_day = next_date_str.split('-')
    year, month, day = year_month_day
    next_date = datetime.datetime(day=int(day),
                                  month=int(month),
                                  year=int(year))
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
windowed_df = df_to_windowed_df(hist, '2014-06-09', '2024-06-09', n=3)


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

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import layers

# model = Sequential([layers.Input((3, 1)),
#                     layers.LSTM(64),
#                     layers.Dense(32, activation='relu'),
#                     layers.Dense(32, activation='relu'),
#                     layers.Dense(1)])

# model.compile(loss='mse',
#               optimizer=Adam(learning_rate=0.001),
#               metrics=['mean_absolute_error'])

# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

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
