# multiple linear regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

data=pd.read_csv("/content/cow_shed_ventilation_dataset.csv")
x=data[["Temperature_C","Humidity_percent","Airflow_mps"]]
y=data[["Ventilation_Index"]]

# create multiple LR model
model = LinearRegression()

# train the model
model.fit(x, y)

# predict
y_pred = model.predict(x)

# result
print("R2 score:", r2_score(y, y_pred))
print("Mean square error:", mean_squared_error(y, y_pred))
print("Intercept:", model.intercept_)
print("Coefficient of ", model.coef_)


# Predictions
y_pred1 = model.predict([[33.21,62.13,1.71]])
print("the prediction found is: ",y_pred1)