#Always check for class imbalance and confusion matrix!!!!!

import pandas as pd
df = pd.read_csv('breast_cancer.csv')
print(df)

print(df.isna().sum())
print(df.duplicated().sum())

df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

columns = ['radius_mean',	'texture_mean',	'perimeter_mean',	'area_mean',	'smoothness_mean',	'compactness_mean',	'concavity_mean',	
'concave points_mean',	'symmetry_mean',	'fractal_dimension_mean',	'radius_se',	'texture_se',	'perimeter_se',	'area_se',	'smoothness_se',	
'compactness_se',	'concavity_se',	'concave points_se',	'symmetry_se',	'fractal_dimension_se',	'radius_worst',	'texture_worst',	
'perimeter_worst',	'area_worst',	'smoothness_worst',	'compactness_worst',	'concavity_worst',	
'concave points_worst',	'symmetry_worst',	'fractal_dimension_worst']

x = df[columns]
y = df['diagnosis']

print(x)
print(y)

print(df['diagnosis'].value_counts())
print(df['diagnosis'].value_counts(normalize=True) * 100)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# XGBoost (no scaling)
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42),
    {
        "n_estimators": [100, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0]
    },
    cv=5, scoring="accuracy", n_jobs=1
)
xgb_grid.fit(x_train, y_train)
print("XGBoost best params:", xgb_grid.best_params_)
best_xgb = xgb_grid.best_estimator_

# --- EVALUATION ---
models = {
    'XGB':    best_xgb
}

for name, model in models.items():
    y_pred  = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"{name} Train Score: {model.score(x_train, y_train):.4f}")
    print(f"{name} Test Score:  {model.score(x_test, y_test):.4f}")
    print(f"\n{name} Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"{name} AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

#SHAP VALUES
import shap
import matplotlib.pyplot as plt

# Keep ID separate before training
patient_ids = df['id']

# Run explainer on ALL data
explainer = shap.TreeExplainer(best_xgb)
shap_values_all = explainer(x)

# Find specific patient
patient_id = 842302
patient_loc = df[df['id'] == patient_id].index[0]

# Plot waterfall for that patient
shap.plots.waterfall(shap_values_all[patient_loc], show=False)
plt.tight_layout()
plt.show()

import joblib
joblib.dump(best_xgb, 'final_xgb_model.pkl')