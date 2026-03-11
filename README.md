# **Air Quality Prediction**

## **GOAL**

The goal of this project is to **predict Air Quality Index (AQI)** from environmental features such as:

* Particulate Matter (PM2.5, PM10)
* Nitrogen Oxide (NO)
* Nitric Dioxide (NO2)
* Carbon Monoxide (CO)
* Sulphur Dioxide (SO2)
* Ozone (O3)

---

## **DATASET**

The dataset can be downloaded from [Kaggle – Air Quality Data in India](https://www.kaggle.com/rohanrao/air-quality-data-in-india).

---

## **PROJECT STEPS**

1. Data Exploration
2. Data Cleaning
3. Data Visualization
4. Feature Selection & Engineering
5. Model Training
6. Model Evaluation

---

## **MODELS USED**

* Linear Regression
* Logistic Regression (not ideal for numeric AQI, included for experimentation)
* K-Nearest Neighbors (KNN) Regressor
* Decision Tree Regressor

---

## **LIBRARIES NEEDED**

* pandas
* numpy
* matplotlib
* seaborn
* sklearn (for models, train-test split, metrics)
* joblib (for saving/loading models)

---

## **PERFORMANCE RESULTS**

| Model                   | Train R² | Test R² | RMSE   |
| ----------------------- | -------- | ------- | ------ |
| Linear Regression       | 0.783    | 0.838   | 50.72  |
| Logistic Regression     | 0.158    | 0.124   | 117.94 |
| KNN Regressor           | 0.887    | 0.876   | 44.29  |
| Decision Tree Regressor | 0.999    | 0.775   | 59.73  |

---

### **CONCLUSION**

* **Linear Regression:** Test R² = 0.838, RMSE = 50.72
* **Logistic Regression:** Test R² = 0.124, RMSE = 117.94 (not suitable for numeric AQI)
* **KNN Regressor:** Test R² = 0.876, RMSE = 44.29 → **Best performance in test prediction accuracy**
* **Decision Tree Regressor:** Test R² = 0.775, RMSE = 59.73 → high train accuracy but some overfitting

**Observations:**

* Regression models (Linear Regression, KNN) show good generalization.
* Logistic Regression is not appropriate for numeric AQI prediction.
* KNN Regressor gave the **lowest RMSE** and good R² on test data → **best model for AQI prediction**.
* Decision Tree achieved **highest train accuracy** (R²=0.999) but lower test R² → likely overfitting.

 **Recommendation:** Use **KNN Regressor** for AQI prediction for better test accuracy and low RMSE.

---

