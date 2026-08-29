# Task 5 — Sales Prediction Using Python

## 📌 Objective
Build a regression model that predicts **product sales** based on
advertising spend across three media channels — TV, Radio, and Newspaper.

## 📂 Dataset
**Advertising dataset** — 200 markets, each with the advertising budget
(in thousand USD) spent on TV, radio, and newspaper, and the resulting
product sales (in thousand units).

| Column | Description |
|---|---|
| `TV` | Advertising budget spent on TV (thousand USD) |
| `Radio` | Advertising budget spent on radio (thousand USD) |
| `Newspaper` | Advertising budget spent on newspaper (thousand USD) |
| `Sales` | Product sales (thousand units) — **target variable** |

## 🛠️ Tech Stack
Python · pandas · scikit-learn · matplotlib · seaborn · Jupyter Notebook

## 🔍 Workflow

1. **Data Loading & EDA** — checked for nulls and duplicates (none found),
   reviewed descriptive statistics, and visualized all feature relationships
   with a pairplot.
2. **Individual Scatter Plots** — Sales vs. TV, Sales vs. Radio, Sales vs.
   Newspaper, each with a fitted regression line.
3. **Correlation Matrix Heatmap** — quantified which channel correlates
   most strongly with Sales.
4. **Train/Test Split** — 80/20 split.
5. **Modeling** — trained and compared:
   - Linear Regression (baseline)
   - Random Forest Regressor
6. **Evaluation** — MAE, RMSE, R² for each model.
7. **Residual Plot** — checked the best model's errors for systematic bias.
8. **Channel Impact Analysis** — compared Linear Regression coefficients and
   Random Forest feature importances to determine which channel drives
   sales the most.

## 📊 Results

| Model | Result |
|---|---|
| Linear Regression | Baseline |
| **Random Forest Regressor** | **Best performer** ✅ |

The **Random Forest Regressor** slightly outperformed Linear Regression on
R² and RMSE — likely because it captures the known interaction effect
between TV and Radio spend (their combined effect on sales is greater than
the sum of their individual linear effects), which a plain linear model
can't represent without manually engineered interaction terms.

Residuals for the best model were randomly scattered around zero with no
clear pattern, indicating the model isn't systematically biased for any
particular sales range.

**Channel impact ranking:** `TV` has by far the highest impact on sales,
followed by `Radio`, with `Newspaper` contributing the least — confirmed
by both the linear regression coefficients and the Random Forest feature
importances. Practically, this suggests advertising budget is best
concentrated on TV (and secondarily Radio), with newspaper spend being the
weakest lever.

## 📁 Files in this folder
- `[YourName]_Task5.ipynb` — full notebook (EDA, modeling, evaluation)
- `Advertising.csv` — dataset used
- `README.md` — this file

## ▶️ How to Run
```bash
pip install pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook
```
Open the notebook and run all cells top to bottom (`Restart Kernel & Run All`).

---
*Part of the Oasis Infobyte Data Science Internship (OIBSIP).*
