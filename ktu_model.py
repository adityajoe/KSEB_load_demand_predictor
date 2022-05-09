import pandas as pd
import numpy as np
from sklearn import linear_model
import joblib
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as cv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import streamlit as st
import seaborn as sns
import plotly.express as px
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
import sklearn.model_selection as cv
import datetime
from datetime import date
import time
raw_data = pd.read_csv("https://raw.githubusercontent.com/adityajoe/KSEB_load_demand_predictor/main/tvm%20all%20data.csv")
final = raw_data.drop(columns = ["Wind speed in kmph"])
date = pd.to_datetime(final["Date_Time"]).dt.date
time = pd.to_datetime(final["Date_Time"]).dt.time
final.insert(1,"Time", time, True)
final.insert(0,"Date",date,True)
final = final.drop(columns = ["Date_Time"])
final["Time"] = final["Time"].map(str)
final["Date"] = final["Date"].map(str)
time_array = np.array(final["Time"])
for i in range(len(time_array)):
  time_array[i] = time_array[i][0:2]
final["Time"] = time_array

new_data = {}
time_demand = []
g = final.groupby("Time")
for time, time_df in g:
    new_data[time] = time_df["Demand in MW"].mean()
print(new_data)
for time1 in np.array(final["Time"]):
  time_demand.append(new_data[time1])
final["time_demand"] = time_demand
final = final.drop(["Time"], axis = 1)

X = final.drop(["Date", "Demand in MW"], axis = 1)
Y = final["Demand in MW"]
day_week = pd.get_dummies(X['Day of the week'],drop_first=True)
X = pd.concat([X, day_week], axis = 1)
X = X.drop(["Day of the week"], axis = 1)
imp = IterativeImputer(max_iter=100, random_state=0,min_value=0.0)
X = imp.fit_transform(X)
X_train, X_test, Y_train, Y_test = cv.train_test_split(X, Y, test_size = 0.33, random_state = 5)
lm = LinearRegression()
lm.fit(X_train, Y_train)
joblib.dump(lm, 'ktu_model1.pkl')
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(random_state=0)
regr.fit(X_train, Y_train)
joblib.dump(regr, 'ktu_mode2.pkl')
