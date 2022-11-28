from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import time
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


df = pd.read_csv("Pearson_corr_Selection.csv")
df1 = pd.read_csv("Pearson_corr_Selection_test.csv")

X = df.drop('Prospect', axis=1)
y = df['Prospect']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=42, stratify=y) 

ms = StandardScaler()
X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)
df1 = ms.transform(df1)

sm = SMOTE(sampling_strategy='not majority', random_state=42, k_neighbors=2, n_jobs=-1)
X_train, y_train = sm.fit_resample(X_train, y_train)
print((Counter(y_train)))
print(sorted(Counter(y_train).items()))

gbc = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01, verbose=100, random_state=2, warm_start=True)

gbc.fit(X_train, y_train)
gbc.score(X_test, y_test)

y_pred = gbc.predict(X_test)
print(f1_score(y_test, y_pred, average='macro'))

prediction = gbc.predict(df1)

submission=pd.read_csv('sample_submission.csv', index_col=0)  
submission['Prospect']=prediction

# submission.to_csv('.csv')