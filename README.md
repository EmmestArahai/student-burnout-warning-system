# 🧠 Student Burnout & Mental Health Warning System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

> An end-to-end Machine Learning web application that analyzes a student's daily habits 
> and predicts their risk of academic burnout — with personalized, actionable recommendations.

🔗 **Live Demo:** [student-burnout-warning-system.streamlit.app](https://student-burnout-warning-system-qs6owlambhnbqxq452qbzl.streamlit.app/)

---

## 📸 App Preview

| Burnout Assessment | EDA Dashboard |
|---|---|
| ![Assessment]((Assessment.png)) | ![Dashboard](!(Dashboard.png)) |

---

## 🎯 Project Overview

Burnout among students is a growing mental health crisis. This project builds an 
**AI-powered early warning system** that:

- Analyzes 20 lifestyle and academic factors
- Predicts burnout risk probability using XGBoost
- Provides personalized action plans with clear explanations
- Compares your habits against 10,000 student records
- Exports a detailed PDF health report

---

## 🏗️ Project Architecture
student-burnout-warning-system/

│

├── data/

│   ├── Student_Productivity_Dataset.csv   ← Raw dataset (10,000 rows)

│   └── featured_data.csv                  ← Cleaned & engineered data

│

├── models/

│   ├── burnout_model.pkl                  ← Trained XGBoost model

│   └── feature_columns.pkl               ← Feature list for inference

│

├── src/

│   ├── data_preprocessing.py             ← Cleaning & encoding

│   ├── feature_engineering.py            ← Custom feature creation

│   ├── train_model.py                    ← Model training pipeline

│   └── predict.py                        ← Inference logic

│

├── app.py                                ← Streamlit web application

├── requirements.txt

└── README.md

---

## 🔬 Technical Deep Dive

### Dataset
- **Source:** Student Productivity Dataset (synthetic, 10,000 rows)
- **Features:** 20 variables covering lifestyle, academic, and psychological metrics
- **Missing Values:** ~1,700 NaN values handled via median/mode imputation

### Feature Engineering
4 custom features were engineered to capture burnout patterns that raw features couldn't:

| Feature | Formula | Insight |
|---|---|---|
| `Sleep_Debt` | `max(0, 6 - sleep_hours)` | Hours below healthy threshold |
| `Digital_Overload` | `screen_time + social_media` | Total digital exposure |
| `Academic_Pressure` | `study_hours × (1 - assignments/100)` | Effort vs. output gap |
| `Lifestyle_Balance` | `physical_activity/7 + sleep` | Recovery capacity |
| `Burnout_Risk_Score` | Weighted composite of above | Single risk indicator |

> 💡 4 out of 5 top features by XGBoost importance were engineered features — validating this approach.

### Target Variable
Burnout_Label = 1 if:
Stress_Level >= 6  OR
Motivation_Level <= 4  OR
Productivity_Score <= 40

This produced a balanced dataset: **56.2% at-risk, 43.8% safe.**

### Model Selection

| Model | Accuracy | ROC-AUC | F1 (At-Risk) |
|---|---|---|---|
| Random Forest | 57% | 0.6249 | 0.68 |
| **XGBoost** ✅ | **60%** | **0.6563** | **0.67** |

XGBoost was selected based on superior ROC-AUC score.

### ML Pipeline
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1))
])
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/student-burnout-warning-system.git
cd student-burnout-warning-system

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 📊 Key Findings

- 🛌 **Sleep** is the strongest positive predictor of student productivity (r = +0.33)
- 📱 **Digital overload** (screen + social media) shows the strongest negative impact (r = -0.25)
- 🏃 **Physical activity** is significantly lower in at-risk students
- 💪 **Motivation level** is the most powerful psychological protective factor
- 📚 Students who study 8+ hours/day without breaks show higher burnout rates

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Visualization | Plotly, Seaborn, Matplotlib |
| Web App | Streamlit |
| PDF Export | ReportLab |
| Deployment | Streamlit Cloud |

---

## ⚠️ Disclaimer

This application is built for **educational only**.  
It is not a clinical or medical tool and should not replace professional mental health advice.  
If you are struggling, please reach out to a qualified counselor or mental health professional.

---

## 📄 License

MIT License — feel free to fork and build upon this project.

---

<p align="center">Built with ❤️ by <a href="https://github.com/EmmestArahai">EmmestArahai</a></p>