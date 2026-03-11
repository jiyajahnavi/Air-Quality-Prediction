import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report, mean_squared_error, mean_absolute_error, r2_score

model1 = joblib.load("model1.pkl")
model2 = joblib.load("model2.pkl")
model3 = joblib.load("model3.pkl")
model4 = joblib.load("model4.pkl")

X_train = joblib.load("x_train.pkl")
y_train = joblib.load("y_train.pkl")
X_test = joblib.load("x_test.pkl")
y_test = joblib.load("y_test.pkl")

#Prediction using all models
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)
pred4 = model4.predict(X_test)


#Accuracy of the model
acc1 = model1.score(X_train, y_train)
acc2 = model2.score(X_train, y_train)
acc3 = model3.score(X_train, y_train)
acc4 = model4.score(X_train, y_train)

print("Train Accuracy (R²):")
print(f"Linear Regression: {acc1:.3f}")
print(f"Logistic Regression: {acc2:.3f}")
print(f"KNN: {acc3:.3f}")
print(f"Decision Tree Regressor: {acc4:.3f}")

#RMSE
print("\nRoot Mean Square Error")
rmse1 = np.sqrt(mean_squared_error(y_test,pred1))
print(f"Linear Regression {rmse1}")
rmse2 = np.sqrt(mean_squared_error(y_test,pred2))
print(f"Logistic Regression {rmse2}")
rmse3 = np.sqrt(mean_squared_error(y_test,pred3))
print(f"KNN {rmse3}")
rmse4 = np.sqrt(mean_squared_error(y_test,pred4))
print(f"Decision Tree Regressor {rmse4}")

# R² Score
r2_1 = r2_score(y_test, pred1)
r2_2 = r2_score(y_test, pred2)
r2_3 = r2_score(y_test, pred3)
r2_4 = r2_score(y_test, pred4)

print("\nR² Score")
print(f"Linear Regression R²: {r2_1:.3f}")
print(f"Logistic Regression R²: {r2_2:.3f}")
print(f"KNN R²: {r2_3:.3f}")
print(f"Decision Tree Regressor R²: {r2_4:.3f}")

models_metrics = {
    "Linear Regression": (acc1, r2_1, rmse1),
    "Logistic Regression": (acc2, r2_2, rmse2),
    "KNN": (acc3, r2_3, rmse3),
    "Decision Tree Regressor": (acc4, r2_4, rmse4)
}

# Best model logic: sort by train R², then test R², then lowest RMSE
best_model = max(models_metrics.items(), key=lambda x: (x[1][1], -x[1][2]))
print(f"\n Best Model: {best_model[0]}")
print(f"Train R²={best_model[1][0]:.3f}, Test R²={best_model[1][1]:.3f}, RMSE={best_model[1][2]:.3f}")

joblib.dump(best_model[0], "best_aqi_model.pkl") 

if best_model == "Linear Regression":
    best_model_obj = model1
elif best_model == "Logistic Regression":
    best_model_obj = model2
elif best_model == "KNN":
    best_model_obj = model3
else:
    best_model_obj = model4

joblib.dump(best_model_obj, "best_aqi_model.pkl")


if not os.path.exists("plots"):
    os.makedirs("plots")

plt.savefig("plots/actual_vs_predicted.png", dpi=300)
plt.show()

# Plot Actual vs Predicted for Best Model

y_pred_best = best_model_obj.predict(X_test)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_best, alpha=0.6, color='teal', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)  # perfect prediction line
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title(f"Actual vs Predicted AQI ({best_model})")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png", dpi=300)
plt.show()