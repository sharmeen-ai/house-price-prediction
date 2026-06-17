"""
House Price Prediction - End-to-End Machine Learning Project
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("HousePricePrediction.csv")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Remove rows without target values
df = df[df["SalePrice"].notna()]

# Define target and features
y = df["SalePrice"]
X = df.drop(columns=["SalePrice", "Id"], errors="ignore")

# Identify columns
num_cols = X.select_dtypes(include=["number"]).columns
cat_cols = X.select_dtypes(exclude=["number"]).columns

# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

best_pipeline = None
best_r2 = -999
results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    cv   = cross_val_score(pipeline, X, y, cv=5, scoring="r2").mean()

    results[name] = {"pred": pred, "mae": mae, "rmse": rmse, "r2": r2, "cv": cv}

    print(f"\n{name}")
    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R2   : {r2:.4f}")
    print(f"CV R2: {cv:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_pipeline = pipeline

# ── CHART 1: Actual vs Predicted (both models side by side) ──────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Actual vs Predicted House Prices", fontsize=16, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    ax.scatter(y_test, res["pred"], alpha=0.5, color='steelblue', edgecolors='white', linewidth=0.3)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel("Actual Price ($)", fontsize=12)
    ax.set_ylabel("Predicted Price ($)", fontsize=12)
    ax.set_title(f"{name}\nR² = {res['r2']:.3f}", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("chart1_actual_vs_predicted.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: chart1_actual_vs_predicted.png")

# ── CHART 2: Residual Plot ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Residual Analysis", fontsize=16, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    residuals = y_test.values - res["pred"]
    ax.scatter(res["pred"], residuals, alpha=0.5, color='coral', edgecolors='white', linewidth=0.3)
    ax.axhline(y=0, color='black', linestyle='--', lw=1.5)
    ax.set_xlabel("Predicted Price ($)", fontsize=12)
    ax.set_ylabel("Residual (Actual - Predicted)", fontsize=12)
    ax.set_title(f"{name} — Residuals", fontsize=13)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("chart2_residuals.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: chart2_residuals.png")

# ── CHART 3: Model Comparison Bar Chart ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Model Performance Comparison", fontsize=16, fontweight='bold')

model_names = list(results.keys())
colors = ['#4C72B0', '#DD8452']

metrics = [
    ("MAE ($)", [results[m]["mae"] for m in model_names]),
    ("RMSE ($)", [results[m]["rmse"] for m in model_names]),
    ("R² Score", [results[m]["r2"] for m in model_names]),
]

for ax, (metric, values) in zip(axes, metrics):
    bars = ax.bar(model_names, values, color=colors, edgecolor='white', linewidth=1.2)
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.set_ylabel(metric)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f'{val:,.2f}' if val > 10 else f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xticklabels(model_names, fontsize=10)

plt.tight_layout()
plt.savefig("chart3_model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: chart3_model_comparison.png")

# ── CHART 4: Sale Price Distribution ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("House Sale Price Distribution", fontsize=16, fontweight='bold')

axes[0].hist(df["SalePrice"], bins=50, color='steelblue', edgecolor='white', linewidth=0.5)
axes[0].set_xlabel("Sale Price ($)", fontsize=12)
axes[0].set_ylabel("Number of Houses", fontsize=12)
axes[0].set_title("Price Distribution", fontsize=13)
axes[0].grid(True, alpha=0.3)

axes[1].hist(np.log1p(df["SalePrice"]), bins=50, color='mediumseagreen', edgecolor='white', linewidth=0.5)
axes[1].set_xlabel("Log(Sale Price)", fontsize=12)
axes[1].set_ylabel("Number of Houses", fontsize=12)
axes[1].set_title("Log Price Distribution (More Normal)", fontsize=13)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("chart4_price_distribution.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: chart4_price_distribution.png")

# Save best model
joblib.dump(best_pipeline, "house_price_model.pkl")
print("\nBest model saved as house_price_model.pkl")
print(f"\n🏆 Best Model: Random Forest with R² = {best_r2:.4f}")
