# Smart Loan Approval System (ML + GenAI)

## Overview
This project is an end-to-end Machine Learning system that predicts whether a loan application should be approved or rejected.  
It also includes a GenAI-style explanation layer that converts model predictions into human-readable insights.

## Tech Stack
- Python
- NumPy, Pandas, Matplotlib
- Scikit-learn
- Random Forest, Logistic Regression
- Joblib
- Git & GitHub

## Project Workflow
1. Data Cleaning & Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Baseline Model: Logistic Regression  
4. Improved Model: Random Forest  
5. Model Evaluation & Comparison  
6. Model Persistence  
7. GenAI-style Explanation Generation  

## Model Performance
| Model | Accuracy |
|-----|---------|
| Logistic Regression | ~77% |
| Random Forest | ~79% |

## Sample Output
Loan Approved because the applicant has good credit history and sufficient income.


## Key Learnings
- Handling real-world data issues (encoding, missing values)
- Building reusable ML pipelines
- Comparing baseline vs advanced models
- Making ML predictions interpretable using GenAI-style explanations

## Future Improvements
- Deploy using Streamlit
- Replace rule-based explanations with LLMs

## How to Run
```bash
cd src
python3 train_model.py

---

## STEP 2: COMMIT README

```bash
git add README.md
git commit -m "Add project README with ML and GenAI overview"
git push
