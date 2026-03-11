import joblib
from sklearn.metrics import accuracy_score,classification_report, mean_squared_error, mean_absolute_error, r2_score

model1 = joblib.load("model1.pkl")
model2 = joblib.load("model2.pkl")
model3 = joblib.load("model3.pkl")
model4 = joblib.load("model4.pkl")

X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)
pred4 = model4.predict(X_test)

