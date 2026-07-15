import pandas as pd
from config import Cleaned_DataPath
df=pd.read_csv(Cleaned_DataPath)
X=df.drop('PlacementStatus',axis=1)
y=df['PlacementStatus']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

#applying scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

import joblib
joblib.dump(scaler, "model/scaler.pkl")

#applying logistic regression
from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression(class_weight='balanced',max_iter=5000)
lr_model.fit(X_train_scaled,y_train)
lr_y_pred=lr_model.predict(X_test_scaled)
print(lr_model.coef_)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print("ACCURACY OF LOGISTIC MODEL = ",accuracy_score(y_test,lr_y_pred))
print("CLASSIFICATION REPORT OF LOGISTIC MODEL = ",classification_report(y_test,lr_y_pred))
print("CONFUSION MATRIX OF LOGISTIC MODEL = ",confusion_matrix(y_test,lr_y_pred))
print("Train Accuracy:", lr_model.score(X_train_scaled,y_train))
print("Test Accuracy:", lr_model.score(X_test_scaled,y_test))

from sklearn.model_selection import GridSearchCV
params={'C':[0.01,0.1,1,10],'penalty':['l1','l2'],'solver':['liblinear','saga']}
grid=GridSearchCV(LogisticRegression(max_iter=5000,class_weight='balanced'),cv=5,param_grid=params,scoring='f1',n_jobs=-1)
grid.fit(X_train_scaled,y_train)
print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)
lr_grid_model = grid.best_estimator_
lr_grid_pred=lr_grid_model.predict(X_test_scaled)
print("ACCURACY OF LOGISTIC MODEL AFTER HYPERPARAMETER TUNING = ",accuracy_score(y_test,lr_grid_pred))
print("CLASSIFICATION REPORT OF LOGISTIC MODEL AFTER HYPERPARAMETER TUNING = ",classification_report(y_test,lr_grid_pred))
print("CONFUSION MATRIX OF LOGISTIC MODEL AFTER HYPERPARAMETER TUNING  = ",confusion_matrix(y_test,lr_grid_pred))
print(lr_grid_model.coef_)
print("Train Accuracy:", lr_model.score(X_train_scaled,y_train))
print("Test Accuracy:", lr_model.score(X_test_scaled,y_test))

from sklearn.feature_selection import SelectKBest,f_classif
selector=SelectKBest(score_func=f_classif,k=10)
X_new=selector.fit_transform(X_train_scaled,y_train)

#to see scores of features
features_score_df=pd.DataFrame({'features':X.columns,'score':selector.scores_})
features_score_df=features_score_df.sort_values(by='score',ascending=False)
print(features_score_df)

selected_features=X.columns[selector.get_support()]
print(selected_features)

X_train_selected=selector.transform(X_train_scaled)
X_test_selected=selector.transform(X_test_scaled)
lr_best_feature_model=LogisticRegression(class_weight='balanced',max_iter=5000,)
lr_best_feature_model.fit(X_train_selected,y_train)
lr_best_feature_pred=lr_best_feature_model.predict(X_test_selected)

print("ACCURACY OF LOGISTIC MODEL AFTER SelectKBest = ",accuracy_score(y_test,lr_best_feature_pred))
print("CLASSIFICATION REPORT OF LOGISTIC MODEL AFTER SelectKBest = ",classification_report(y_test,lr_best_feature_pred))
print("CONFUSION MATRIX OF LOGISTIC MODEL AFTER SelectKBest  = ",confusion_matrix(y_test,lr_best_feature_pred))

from sklearn.feature_selection import RFE
rfe=RFE(estimator=lr_model,n_features_to_select=10)
X_train_rfe=rfe.fit_transform(X_train_scaled,y_train)
X_test_rfe=rfe.transform(X_test_scaled)

rfe_df=pd.DataFrame({'features':X.columns,
                     'rank':rfe.ranking_})
rfe_df=rfe_df.sort_values(by='rank')
print(rfe_df)
selected_features_rfe=X.columns[rfe.support_]
print(selected_features_rfe)

#TRAIN MODEL AFTER APPLYING RFE FEATURE SELECTION TECHNIQUES
lr_model_rfe=LogisticRegression(class_weight='balanced',max_iter=5000)
lr_model_rfe.fit(X_train_rfe,y_train)
lr_rfe_predict=lr_model_rfe.predict(X_test_rfe)

print("ACCURACY OF LOGISTIC MODEL AFTER RFE = ",accuracy_score(y_test,lr_rfe_predict))
print("CLASSIFICATION REPORT OF LOGISTIC MODEL AFTER RFE = ",classification_report(y_test,lr_rfe_predict))
print("CONFUSION MATRIX OF LOGISTIC MODEL AFTER RFE = ",confusion_matrix(y_test,lr_rfe_predict))

# saving logistic regression tuned as our best model
import joblib
joblib.dump(lr_grid_model,"model/best_model.pkl")

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_prob = lr_grid_model.predict_proba(X_test_scaled)[:,1]

auc_score = roc_auc_score(y_test, y_prob)

print("ROC-AUC Score =", auc_score)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression Tuned")
plt.legend()
plt.savefig("images/roc_auc_lr.png",dpi=300,bbox_inches='tight')
print("image saved successfully")

from utils import save_result
save_result(
    "Logistic Regression Baseline",
    y_test,
    lr_y_pred
)
save_result(
    "Logistic Regression Tuned",
    y_test,
    lr_grid_pred
)
save_result(
    "Logistic Regression SelectKBest",
    y_test,
    lr_best_feature_pred
)
save_result(
    "Logistic Regression RFE",
    y_test,
    lr_rfe_predict
)
