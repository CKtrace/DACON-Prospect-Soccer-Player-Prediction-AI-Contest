import optuna
from optuna import Trial
import pandas as pd
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings(action='ignore')

df = pd.read_csv("Pearson_corr_Selection.csv")
X = df.drop('Prospect', axis=1)
y = df['Prospect']

X_train, val_X, y_train, val_y = train_test_split(X, y, test_size = 0.05, random_state=2, stratify=y)

# sm = BorderlineSMOTE(sampling_strategy='auto',random_state=42, k_neighbors=3, m_neighbors=11, n_jobs=-1)
# X_train, y_train = sm.fit_resample(X_train, y_train)
# print((Counter(y_train)))
# print(sorted(Counter(y_train).items()))

print(X_train.shape, y_train.shape)

sd = StandardScaler()
X_train = sd.fit_transform(X_train)
val_X = sd.transform(val_X)

def objectiveXGB(trial: Trial, X, y):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'max_depth' : trial.suggest_int('max_depth', 8, 16),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
        'gamma' : trial.suggest_int('gamma', 1, 3),
        'learning_rate' : trial.suggest_float('learning_rate', 0.001, 0.2),
        'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        'nthread' : -1,
        'tree_method' : 'gpu_hist',
        'predictor' : 'gpu_predictor',
        'lambda' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha' : trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample' : trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
        'random_state' : trial.suggest_int("random_state", 1, 1000),
        'n_jobs' : 8
    }
    model = XGBClassifier(**param)
    xgb_model = model.fit(X_train, y_train, verbose=True)
    
    score = accuracy_score(xgb_model.predict(val_X), val_y)
    
    return score

study = optuna.create_study(direction='maximize', sampler=TPESampler())

study.optimize(lambda trial : objectiveXGB(trial, X, y), n_trials = 100)

print('Best trial : score {}, \nparams {}'.format(study.best_trial.value, study.best_trial.params))


