# cow-shed-ml_linear-regression
devloping ml (simple linear regression )model for cow shed vantilation , testing and training the data set.
<br>
code for simple linear regression 
<br>
#STEP 1 IMPORTING THE NESSESSARY LIBRARY
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as pit

#STEP 2 : IMPORTING OF DATA
x= np.array ((60,80,100,120,140)).reshape(-1,1)
y = np.array ([230,210,190,170,155])

#STEP 3: DEVELOP MODEL
model= LinearRegression()
model.fit(x,y)
y_pred= model.predict(x)


#STEP 4: MODEL ESTIMATION (METRICS)
mse= mean_squared_error(y,y_pred)
r2= r2_score(y,y_pred)
rmse = np.sqrt (mse)

#STEP 5 : GET THE RESULTT
print("Slope (n):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)
