# diabetes-model-training

# 🤖 Diabetes Prediction AI (SDG 3 - Good Health)

## 🌍 Project Overview
This machine learning project supports **UN SDG 3: Good Health and Well-being** by predicting the likelihood of diabetes based on patient data.

## 📊 Dataset
- **Source**: Pima Indians Diabetes Dataset
- **Fields**: Glucose, BMI, Age, Blood Pressure, etc.

## 🧠 ML Approach
- **Type**: Supervised Learning
- **Model**: Decision Tree Classifier
- **Library**: scikit-learn

## 📈 Results
- **Accuracy**: ~75% (depends on train/test split)
- **Tools Used**: Python, pandas, scikit-learn, seaborn, matplotlib

## ⚖️ Ethical Reflection
- Dataset may lack diversity across ethnic backgrounds
- False negatives could be risky for real health applications
- Important to use this model as **decision support**, not replacement for doctors

## 📷 Screenshots
See the `screenshots/` folder for evaluation outputs.

---

## 🚀 How to Run
```bash
pip install pandas scikit-learn matplotlib seaborn
python diabetes_model.py
