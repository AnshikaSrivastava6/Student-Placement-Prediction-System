import pandas as pd
from config import Cleaned_DataPath
df=pd.read_csv(Cleaned_DataPath)
X=df.drop('PlacementStatus',axis=1)
y=df['PlacementStatus']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)

#applying random forest to the dataset
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(random_state=42,class_weight='balanced')
rf_model.fit(X_train,y_train)
rf_pred=rf_model.predict(X_test)

print("ACCURACY OF RANDOM FOREST MODEL = ",accuracy_score(y_test,rf_pred))
print("CLASSIFICATION REPORT OF RANDOM FOREST MODEL = ",classification_report(y_test,rf_pred))
print("CONFUSION MATRIX OF RANDOM FOREST MODEL = ",confusion_matrix(y_test,rf_pred))

from sklearn.model_selection import GridSearchCV
params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=params,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print("BEST PARAMETERS :", grid.best_params_)
print("BEST SCORE :", grid.best_score_)

rf_grid_model=grid.best_estimator_
rf_grid_model.fit(X_train,y_train)
rf_grid_pred=rf_grid_model.predict(X_test)

print("ACCURACY OF RANDOM FOREST MODEL AFTER TUNING = ",accuracy_score(y_test,rf_grid_pred))
print("CLASSIFICATION REPORT OF RANDOM FOREST MODEL AFTER TUNING = ",classification_report(y_test,rf_grid_pred))
print("CONFUSION MATRIX OF RANDOM FOREST MODEL AFTER TUNING = ",confusion_matrix(y_test,rf_grid_pred))

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_prob = rf_grid_model.predict_proba(X_test)[:,1]

auc_score = roc_auc_score(y_test, y_prob)

print("ROC-AUC Score =", auc_score)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest Tuned")
plt.legend()
plt.show()

from utils import save_result

save_result(
    "Random Forest Tuned",
    y_test,
    rf_grid_pred
)
save_result(
    "Random Forest Baseline",
    y_test,
    rf_pred
)