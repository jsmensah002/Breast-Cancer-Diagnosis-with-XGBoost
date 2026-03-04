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

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# XGBoost Classifier
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    {
        "n_estimators": [100, 300],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    },
    cv=5,
    scoring="accuracy",
    n_jobs=1
)
xgb_grid.fit(x_train, y_train)
print("XGBoost best params:", xgb_grid.best_params_)

best_xgb = xgb_grid.best_estimator_
print('Train R2:',best_xgb.score(x_train,y_train))
print('Test R2:',best_xgb.score(x_test,y_test))

y_pred_xgb = best_xgb.predict(x_test)

from sklearn.metrics import recall_score, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix

print("xgb Recall:", recall_score(y_test, y_pred_xgb))
print("xgb Precision:", precision_score(y_test, y_pred_xgb))
print("xgb Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))