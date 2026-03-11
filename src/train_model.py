from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd 
import joblib

prepareddata = pd.read_csv(r"D:\Machine Learning\Air Quality Prediction\data\prepared_data.csv")

X = prepareddata[['PM2.5','PM10','NO','NO2','CO','SO2','O3']]
y = prepareddata['AQI']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model1 = LinearRegression()
model2 = LogisticRegression()
model3 = KNeighborsRegressor()
model4 = DecisionTreeRegressor()

model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
model3.fit(X_train,y_train)
model4.fit(X_train,y_train)

joblib.dump(model1, "model1.pkl")
joblib.dump(model2, "model2.pkl")
joblib.dump(model3, "model3.pkl")
joblib.dump(model4, "model4.pkl")

joblib.dump(X_test, "x_test.pkl")
joblib.dump(y_test, "y_test.pkl")
joblib.dump(X_train, "x_train.pkl")
joblib.dump(y_train, "y_train.pkl")