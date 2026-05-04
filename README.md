# loan-risk-prediction
Machine learning project for predicting loan default using classification models

## Project Overview

This project focuses on predicting whether a borrower will default on a loan using machine learning techniques. Financial institutions face significant risks when issuing loans, and accurate prediction of loan repayment behavior helps in minimizing financial losses and improving decision-making.

## Objective

The main objective of this project is to classify borrowers into:

High-risk (default)

Low-risk (non-default)

## The target variable used is:

repay_fail (1 = default, 0 = repaid)

## Dataset Description

Total Records: 38,479

Features: 24 (23 independent + 1 target)

## Data Types:

Numerical (e.g., loan amount, income, interest rate)

Categorical (e.g., home ownership, loan purpose, verification status)

## Data Preprocessing

The following steps were applied:

### Handling Missing Values
Used median imputation for numerical features
### Feature Scaling
Applied MinMaxScaler for normalization
### Encoding
Ordinal Encoding (for ordered features like loan term)
One-Hot Encoding (for categorical features)

## Machine Learning Models Used

The following models were implemented and compared:

- Logistic Regression

- Decision Tree (Gini & Entropy)

- Support Vector Machine (Linear, RBF, Polynomial, Sigmoid)

- Random Forest

## Hyperparameter Tuning

Used GridSearchCV for optimization

## Tuned parameters such as:

-  max_depth, min_samples_split (Decision Tree)

-  C, kernel, gamma (SVM)

-  n_estimators, depth (Random Forest)
##  Evaluation Metrics

To evaluate model performance, the following metrics were used:

* **Accuracy** 
* **Recall** 
* **Precision** 
* **F1-Score** 

 In this project, **Recall is prioritized**, as missing a defaulter (false negative) can lead to financial loss.
##  Model Performance Results

| Model                   | Accuracy   | Recall     | Precision | F1-Score   |
| ----------------------- | ---------- | ---------- | --------- | ---------- |
| Decision Tree (Entropy) | **1.0000** | 0.9998     | 1.0000    | 0.9999     |
| Decision Tree (Gini)    | **1.0000** | 0.9998     | 1.0000    | 0.9999     |
| Logistic Regression     | 0.9999     | 0.9992     | 1.0000    | 0.9996     |
| Linear SVM (LSVC)       | **1.0000** | 0.9998     | 1.0000    | 0.9999     |
| RBF SVM (RSVC)          | **1.0000** | 0.9997     | 1.0000    | 0.9999     |
| Sigmoid SVM (SSVC)      | 0.9825     | 0.9414     | 0.9426    | 0.9419     |
| Polynomial SVM (PSVC)   | **1.0000** | **0.9999** | 1.0000    | **1.0000** |
| Random Forest (RF)      | 0.9998     | 0.9989     | 1.0000    | 0.9994     |

##  Results Analysis

* Most models achieved **extremely high performance (~99–100%)**, indicating strong predictive capability.
* **Polynomial SVM (PSVC)** achieved the best overall performance with perfect F1-score and highest recall.
* **Decision Trees and Linear SVM** also performed exceptionally well with near-perfect scores.
* **Random Forest** demonstrated strong generalization and stability across metrics.
* **Sigmoid SVM (SSVC)** showed comparatively lower performance, indicating it is less suitable for this dataset.

👉 Overall, the results demonstrate that machine learning models can effectively predict loan default risk with high accuracy and reliability.
# Loan Default Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting whether a borrower will default on a loan using machine learning techniques. Financial institutions face significant risks when issuing loans, and accurate prediction of loan repayment behavior helps in minimizing financial losses and improving decision-making.

---

## 🎯 Objective

The main objective of this project is to classify borrowers into:

* **High-risk (default)**
* **Low-risk (non-default)**

The target variable used is:

* `repay_fail` (1 = default, 0 = repaid)

---

## 📊 Dataset Description

* Total Records: **38,479**
* Features: **24 (23 independent + 1 target)**
* Data Types:

  * Numerical (e.g., loan amount, income, interest rate)
  * Categorical (e.g., home ownership, loan purpose, verification status)

---

## ⚙️ Data Preprocessing

The following steps were applied:

* **Handling Missing Values**

  * Used median imputation for numerical features

* **Feature Scaling**

  * Applied MinMaxScaler for normalization

* **Encoding**

  * Ordinal Encoding (for ordered features like loan term)
  * One-Hot Encoding (for categorical features)

---

## 🤖 Machine Learning Models Used

The following models were implemented and compared:

* Logistic Regression
* Decision Tree (Gini & Entropy)
* Support Vector Machine (Linear, RBF, Polynomial, Sigmoid)
* Random Forest

---

## 🔧 Hyperparameter Tuning

* Used **GridSearchCV** for optimization
* Tuned parameters such as:

  * max_depth, min_samples_split (Decision Tree)
  * C, kernel, gamma (SVM)
  * n_estimators, depth (Random Forest)

---

## 📈 Evaluation Metrics

Models were evaluated using:

* Accuracy
* Recall

👉 Recall was prioritized because detecting loan defaulters is critical in financial risk management.

---

##  Overfitting Handling

* Compared training and testing performance
* Used cross-validation
* Controlled model complexity

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy


---

## 🚀 Future Improvements

* Use larger and more diverse datasets
* Apply deep learning models
* Integrate decision-focused learning approaches

---

##  Author

**Rahat Idrees**
MSc Artificial Intelligence (Ongoing)
Dublin, Ireland

