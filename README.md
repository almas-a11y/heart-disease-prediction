# Heart Disease Risk Prediction

Predicting whether a patient has heart disease using regularized logistic regression, built for COSC 370 (Machine Learning) at Whittier College.

Heart disease is one of the leading causes of death worldwide. This project builds and compares four logistic regression models on the UCI Heart Disease dataset to see which one best predicts whether a patient has heart disease based on their clinical measurements.

## Results

| Model | AUC-ROC | Accuracy | Recall |
|---|---|---|---|
| Ridge (L2) | **0.9000** | 0.7222 | 0.4167 |
| **Lasso (L1)** | 0.8931 | **0.8519** | **0.9167** |
| Logistic Regression (baseline) | 0.8917 | **0.8519** | **0.9167** |
| Polynomial + Ridge | 0.8764 | 0.8148 | 0.7917 |

Ridge (L2) has the highest AUC-ROC, but its recall is by far the worst of the four (0.4167, meaning it misses well over half the actual disease cases). Since recall was weighted heavily for this problem, **Lasso (L1) is the practical winner**: nearly identical AUC-ROC to Ridge, and more than double the recall. Every regularized model beat the plain baseline on AUC-ROC, showing regularization helps even on a small dataset, but for a screening tool the recall gap is what actually matters. The strongest predictors across the interpretable models were `thalach` (max heart rate), `cp` (chest pain type), and `ca` (number of major vessels), which lines up with established clinical risk factors.

## Dataset

UCI Heart Disease dataset (pulled directly from OpenML, data ID 53), 270 patient records from the Cleveland Clinic Foundation with 13 clinical and demographic features (age, sex, chest pain type, resting blood pressure, cholesterol, and others). The original 0–4 severity target was simplified to a binary label: 0 = no disease, 1 = disease present.

## Approach

- **Preprocessing:** missing values filled with the column median, an 80/20 stratified train/test split (216 train / 54 test), and feature scaling with `StandardScaler` (fit on training data only, to avoid leakage)
- **Models compared:**
  - Logistic Regression (no regularization), used as the baseline
  - Ridge (L2), penalizes large coefficients and keeps all features
  - Lasso (L1), can zero out weak features for automatic feature selection
  - Polynomial (degree 2) + Ridge, expands to 91 features to test whether non-linear interactions help
- **Tuning:** 5-fold stratified `GridSearchCV` over `C = [0.001, 0.01, 0.1, 1, 10, 100]`, scored on AUC-ROC
- **Evaluation:** accuracy, precision, recall, AUC-ROC, confusion matrices, and ROC curves on the held-out test set. Recall was weighted heavily since missing a sick patient (a false negative) is far more costly than a false alarm in a medical setting.

## Tech Stack

Python, scikit-learn, pandas, NumPy, matplotlib, seaborn. Run as a Jupyter/Colab notebook.

## Running It

```bash
pip install -r requirements.txt
jupyter notebook heart_disease_prediction.ipynb
```

The notebook pulls the dataset directly from OpenML, so no manual download is needed.

## Future Work

- More data, since 270 samples is small, especially for the 91-feature polynomial model
- Neural networks, which could pick up non-linear boundaries logistic regression can't
- Handling class imbalance (e.g. SMOTE or class-weighted loss) to push recall higher for the disease class
- Additional features such as imaging data, genetic information, or lifestyle factors

## Author

Almas Waseem, done independently for COSC 370.
