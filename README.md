 # 📊 Customer Churn Prediction & Retention Analytics System

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Sklearn](https://img.shields.io/badge/Scikit--Learn-1.0-orange)
![Pandas](https://img.shields.io/badge/Pandas-3.0-green)
![SQL](https://img.shields.io/badge/SQL-SQLite-lightgrey)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 🧠 Business Problem

A telecom company is losing customers every month.
Every churned customer = lost revenue.

This project builds a complete **end-to-end ML system** to:
- Predict which customers will churn
- Identify high-risk customer segments
- Quantify revenue impact
- Recommend retention strategies

---
LIVE DEMO
https://customer-churn-prediction-ltyxn3lqbrhqgt9thaxggg.streamlit.app

---


## 📁 Project Structure
customer-churn-prediction/
""
│

├── data/

│   ├── raw/                  # Original dataset

│   └── processed/            # Cleaned dataset

│

├── notebooks/

│   ├── 01_data_exploration.ipynb

│   ├── 02_sql_analysis.ipynb

│   ├── 03_machine_learning.ipynb

│   └── 04_business_insights.ipynb

│

├── models/

│   ├── logistic_regression.pkl

│   ├── random_forest.pkl

│   └── scaler.pkl

│

├── sql/

│   └── analysis_queries.sql

│

├── reports/

│   └── figures/              # All charts & visualizations

│

├── streamlit_app/

│   └── app.py

│

└── README.md

---

## 📊 Dataset

**IBM Telco Customer Churn Dataset**
- 7,043 customers
- 21 features
- Target variable: `Churn` (Yes/No)

### Key Columns
| Column | Description |
|---|---|
| tenure | Months customer has stayed |
| MonthlyCharges | Amount charged per month |
| Contract | Type of contract |
| InternetService | Type of internet service |
| PaymentMethod | How customer pays |
| Churn | Whether customer left |
""

---

## 🔄 Project Workflow
Data Loading → Cleaning → EDA → SQL Analysis → ML Models → Business Insights

---

## 🔍 Key Findings

### EDA Insights
- Overall churn rate: **26.54%**
- Churned customers stay only **18 months** on average
- Loyal customers stay **37 months** on average
- Churned customers pay **$74/month** vs **$61/month** for loyal ones

### SQL Insights
- **60.37%** churn rate among month-to-month fiber optic customers paying by electronic check
- Month-to-month contracts churn at **43%**
- Two year contracts churn at only **3%**

### Business Impact
- Monthly revenue lost to churn: **$139,000+**
- Annual revenue lost: **$1.6M+**
- Retaining just 10% of churners saves **$167,000/year**

---

## 🤖 Machine Learning Models

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 80.70% | 0.8417 |
| Random Forest | 80.34% | 0.8405 |

### Top Churn Predictors
1. **Tenure** — newer customers churn more
2. **Total Charges** — higher bills = higher churn
3. **Monthly Charges** — pricing sensitivity
4. **Fiber Optic Internet** — service quality issues
5. **Electronic Check** — payment friction

---

## 💡 Business Recommendations

1. **Promote long-term contracts** — reduce month-to-month churn by 15%
2. **Fix fiber optic experience** — investigate quality & pricing
3. **Incentivize auto-pay** — convert electronic check users
4. **New customer onboarding** — first 12 months are critical
5. **Senior citizen retention plans** — dedicated support & pricing

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Database | SQLite, SQL |
| Deployment | Streamlit |
| IDE | VS Code |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git

# Go into the folder
cd customer-churn-prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

---

## 🚀 Run the Project

```bash
# Run Streamlit app
streamlit run streamlit_app/app.py
```

---

## 📈 Results Summary

- ✅ Cleaned and analyzed **7,043 customer records**
- ✅ Built **10+ EDA visualizations**
- ✅ Wrote **5+ SQL business queries**
- ✅ Trained **2 ML models** with 80%+ accuracy
- ✅ Identified **$1.6M annual revenue** at risk
- ✅ Generated **5 actionable business recommendations**


---

## 📬 Contact

- GitHub: [@yourusername](https://github.com/deshaish1602)
  
 
 
