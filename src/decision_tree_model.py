import pandas as pd
from config import Cleaned_DataPath
df=pd.read_csv(Cleaned_DataPath)
X=df.drop('PlacementStatus',axis=1)
y=df['PlacementStatus']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train,y_train)
dt_pred=dt_model.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print("ACCURACY OF DECISION TREE MODEL = ",accuracy_score(y_test,dt_pred))
print("CLASSIFICATION REPORT OF DECISION TREE MODEL =",classification_report(y_test,dt_pred))
print("CONFUSION MATRIX OF DECISION MODEL =",confusion_matrix(y_test,dt_pred))
#TO CHECK OVERFITTING
print("TRAINING ACCURACY  = ",dt_model.score(X_train,y_train))
print("TESTING ACCURACY = ",dt_model.score(X_test,y_test))

from sklearn.model_selection import GridSearchCV
params = { 'criterion':['gini','entropy'], 'max_depth':[3,5,7,10,15,20], 'min_samples_split':[2,5,10,20], 'min_samples_leaf':[1,2,5,10]}
grid=GridSearchCV(DecisionTreeClassifier(random_state=42),param_grid=params,cv=5,n_jobs=-1,scoring='f1')
grid.fit(X_train,y_train)
print("BEST PARAMETERS : ",grid.best_params_)
print("BEST SCORE : ",grid.best_score_)

dt_grid_model=grid.best_estimator_
dt_grid_model.fit(X_train,y_train)
dt_grid_pred=dt_grid_model.predict(X_test)

print("ACCURACY OF DECISION TREE MODEL AFTER HYPERPARAMETER TUNING = ",accuracy_score(y_test,dt_grid_pred))
print("CLASSIFICATION REPORT OF DECISION TREE MODEL AFTER HYPERPARAMETER TUNING  =",classification_report(y_test,dt_grid_pred))
print("CONFUSION MATRIX OF DECISION MODEL AFTER HYPERPARAMETER TUNING =",confusion_matrix(y_test,dt_grid_pred))

from utils import save_result

save_result(
    "Decision Tree Baseline",
    y_test,
    dt_pred
)
save_result(
    "Decision Tree Tuned",
    y_test,
    dt_grid_pred
)

print("SAVED")