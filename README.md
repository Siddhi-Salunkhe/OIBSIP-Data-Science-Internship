# Task 3 — Car Price Prediction with Machine Learning

## 📌 Objective
Build a regression model that predicts the **selling price of a used car**
based on features such as brand, age, mileage, fuel type, and transmission.

## 📂 Dataset
**Vehicle dataset from CarDekho** — 301 used-car listings scraped from
cardekho.com, containing the car's name, manufacturing year, selling price,
current showroom price, kilometers driven, fuel type, seller type,
transmission, and number of previous owners.

| Column | Description |
|---|---|
| `Car_Name` | Model name of the car |
| `Year` | Year of manufacture |
| `Selling_Price` | Price the current owner is asking (lakh INR) — **target variable** |
| `Present_Price` | Current ex-showroom price of an equivalent new car (lakh INR) |
| `Kms_Driven` | Total kilometers driven |
| `Fuel_Type` | Petrol / Diesel / CNG |
| `Seller_Type` | Dealer / Individual |
| `Transmission` | Manual / Automatic |
| `Owner` | Number of previous owners |

## 🛠️ Tech Stack
Python · pandas · scikit-learn · matplotlib · seaborn · Jupyter Notebook

## 🔍 Workflow

1. **Data Cleaning** — checked for nulls (none found), removed 2 duplicate
   listings, normalized categorical text.
2. **Feature Engineering** —
   - `Car_Age` = 2020 − `Year`
   - `Brand` = first word of `Car_Name`
   - Dropped `Car_Name` and `Year` (redundant after engineering).
3. **Exploratory Data Analysis** —
   - Distribution of selling prices (right-skewed).
   - Selling price vs. fuel type (Diesel resells highest).
   - Selling price vs. car age (clear depreciation trend).
   - Correlation heatmap (`Present_Price` most strongly correlated with `Selling_Price`).
4. **Encoding** — one-hot encoded `Fuel_Type`, `Seller_Type`, `Transmission`, `Brand`.
5. **Train/Test Split** — 80/20 split.
6. **Modeling** — trained and compared:
   - Linear Regression (baseline)
   - Random Forest Regressor
7. **Evaluation** — MAE, RMSE, R² for each model.
8. **Feature Importance** — identified top drivers of price for the best model.

## 📊 Results

| Model | R² Score |
|---|---|
| Linear Regression | Baseline |
| **Random Forest Regressor** | **Best performer** ✅ |

The **Random Forest Regressor** outperformed Linear Regression on all metrics
(higher R², lower MAE/RMSE). This is because the relationship between a car's
age, mileage, and price isn't purely linear — depreciation curves flatten
over time, and a tree-based ensemble captures that non-linearity without
needing manually engineered interaction terms.

**Top predictive features:** `Present_Price`, `Car_Age`, and `Kms_Driven`
had the greatest influence on predicted selling price — a car's original
showroom price anchors its resale value more than any other single factor.

## 📁 Files in this folder
- `[YourName]_Task3.ipynb` — full notebook (EDA, modeling, evaluation)
- `car_data.csv` — dataset used
- `README.md` — this file

## ▶️ How to Run
```bash
pip install pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook
```
Open the notebook and run all cells top to bottom (`Restart Kernel & Run All`).

---
*Part of the Oasis Infobyte Data Science Internship (OIBSIP).*
