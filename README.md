Brief Overview:
- Classifying breast cancer tumors as Benign or Malignant using the UCI Breast Cancer dataset. Hyperparameter tuning was performed on all models using GridSearchCV to find the best parameters.

Model: XGBoost Classifier

Results:
- Train Score: 1.000 
- Test Score: 0.974
- Recall: 0.970 || Precision: 0.970 || Accuracy: 0.970
- Confusion Matrix: 70 True Negatives, 41 True Positives, 1 False Positive, 2 False Negatives

Key Insight:
- XGBoost shows signs of overfitting (Train R2: 1.0 vs Test R2: 0.9737), suggesting other models such as Logistic Regression, SVC, and Random Forest could be compared against XGBoost to find a more generalized solution.
- Recall is the most critical metric in medical diagnosis. A false negative (predicting Benign when actually Malignant) is far more dangerous than a false positive. With 2 false negatives recorded, comparing XGBoost against other models such as Logistic Regression, SVC, and Random Forest may reduce this number further. If false negatives do not drop with other models, further hyperparameter tuning of XGBoost would be required.

Deployment:
- The XGBoost model was packaged into a Flask app and deployed to AWS EC2.
