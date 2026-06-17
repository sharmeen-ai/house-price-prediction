# 🏠 House Price Prediction

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

> Predicts house sale prices using an end-to-end ML pipeline.
> Compares **Linear Regression** vs **Random Forest** and saves the best model.

---

## 📊 Results

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | ~$22,000 | ~$35,000 | ~0.78 |
| **Random Forest** ✅ | **~$15,000** | **~$25,000** | **~0.89** |

Random Forest outperformed Linear Regression and was saved as the final model.

---

## 🧠 What This Project Does

- Loads and cleans real estate data (2,919 records, 13 features)
- Handles missing values using median/mode imputation
- Encodes categorical variables with OneHotEncoder
- Trains and compares 2 ML models
- Evaluates using MAE, RMSE, R², and 5-fold Cross Validation
- Saves the best model using joblib

---

## 🛠️ Tools Used

- Python, Pandas, NumPy
- scikit-learn (Pipeline, RandomForest, LinearRegression)
- joblib (model saving)

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python house_price_prediction_ml_project.py
```

---

## 📁 Files

| File | Description |
|---|---|
| `house_price_prediction_ml_project.py` | Main ML script |
| `HousePricePrediction.csv` | Dataset |
| `house_price_model.pkl` | Saved best model |
| `requirements.txt` | Dependencies |

---

## 👤 Author
Available for freelance work → [Hire me on Upwork](YOUR_UPWORK_LINK)

Copy
