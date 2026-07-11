# 📈 Apple Stock Price Prediction — Machine Learning

Predicting Apple (AAPL) stock closing prices using Support Vector Regression (SVR) with hyperparameter tuning, compared against a Linear Regression baseline.

## 📊 Project Workflow

1. **Data Loading & Cleaning** — Historical Apple stock data (Open, High, Low, Close, Volume)
2. **Exploratory Data Analysis (EDA)** — Shape, data types, missing values, duplicates & statistical summary
3. **Feature Engineering**
   - 7-Day & 30-Day Moving Averages
   - Daily Returns calculation
   - Outlier detection using IQR method (extreme volatility days)
4. **Data Visualization** — 5 detailed plots:
   - Closing price trend over time
   - Daily returns distribution (Histogram + KDE)
   - Violin plot of yearly price distribution (2020–2024)
   - Pair plot of stock features
   - Correlation heatmap
5. **Model Building**
   - **SVR (RBF Kernel)** with GridSearchCV hyperparameter tuning (C & Gamma)
   - **Linear Regression** as baseline for comparison
6. **Evaluation** — MAE, MSE, RMSE & R² Score
7. **Results Visualization** — Actual vs Predicted price comparison

## 🛠️ Tech Stack

- **Language:** Python
- **Environment:** Jupyter Notebook (Anaconda)
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn

## 🚀 How to Run

1. Clone this repository
2. Install requirements: `pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn`
3. Open `ApplStock_Prediction.ipynb` in Jupyter Notebook
4. Run all cells

## 👩‍💻 Author

**Ashna Usmani**
GitHub: [@ashnausmanii](https://github.com/ashnausmanii)
