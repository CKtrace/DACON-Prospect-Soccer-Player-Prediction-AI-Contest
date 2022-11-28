from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from collections import Counter


def xgb_oof(model, X, y, X_train, y_train, X_test, y_test, df1, fold_cnt):
    train_pred = np.zeros(shape=(X_train.shape[0], len(y_train.unique())))
    test_pred = np.zeros(shape=(X_test.shape[0], len(y_test.unique())))
    sub_pred = np.zeros(shape=(df1.shape[0], len(y_test.unique())))

    
    stk = StratifiedKFold(n_splits=fold_cnt, shuffle=True, random_state=2)

    for train_index, test_index in stk.split(X, y):
        X_train_, X_test_ = X[train_index], X[test_index]
        y_train_, y_test_ = y[train_index], y[test_index]
        
        model.fit(X_train_, y_train_)
        fold_pred = model.predict(X_test_)
        print(accuracy_score(fold_pred, y_test_))

        train_pred_ = model.predict_proba(X_train)
        test_pred_ = model.predict_proba(X_test)
        sub_pred_ = model.predict_proba(df1)
        
        train_pred += train_pred_/stk.n_splits
        test_pred += test_pred_/stk.n_splits
        sub_pred += sub_pred_/stk.n_splits
    
    
    return sub_pred
        

df = pd.read_csv("Pearson_corr_Selection.csv")
df1 = pd.read_csv("Pearson_corr_Selection_test.csv")
X = df.drop('Prospect', axis=1)
y = df['Prospect']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state=2, stratify=y)


print(X_train.shape, y_train.shape)


params = {'n_estimators': 932,
          'max_depth': 12, 
          'min_child_weight': 83,
          'gamma': 3, 
          'learning_rate': 0.0055532997985024354,
          'colsample_bytree': 0.5, 
          'lambda': 0.005838201587944736, 
          'alpha': 0.7865989548475596, 
          'subsample': 0.6, 
          'random_state': 3,
          'nthread' : -1,
          'tree_method' : 'gpu_hist',
          'predictor' : 'gpu_predictor',
          'n_jobs' : 8}

model = XGBClassifier(**params)

X = np.array(X)
y= np.array(y)

a = xgb_oof(model, X, y, X_train, y_train, X_test, y_test, df1, 5)

submission = []

for item in a:
    if item[0] > item[1]:
        submission.append(0)
    else:
        submission.append(1)
        
print(submission[:5])

submis=pd.read_csv('sample_submission.csv', index_col=0)  
submis['Prospect']=submission

# submis.to_csv('.csv')