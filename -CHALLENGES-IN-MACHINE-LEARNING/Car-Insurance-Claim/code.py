# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)
df.head(5)
print(df.info())
cols = ['INCOME', 'HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']
df[cols] = df[cols].replace({'\$': '', ',': ''}, regex=True)
df.head()

X=df.iloc[:,:-1]
y=df.iloc[:,-1]
count=y.value_counts()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=6)
# Code ends here


# --------------
# Code starts here
cols = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']
X_train[cols] = X_train[cols].apply(pd.to_numeric, errors='coerce')

X_test[cols] = X_test[cols].apply(pd.to_numeric, errors='coerce')

print(X_train.isnull().sum())
print(X_test.isnull().sum())
print(X_train.dtypes)



# Code ends here


# --------------
# Code starts here
X_train.dropna(subset = ['YOJ','OCCUPATION'],inplace=True)

X_test.dropna(subset = ['YOJ','OCCUPATION'],inplace=True)

y_train=y_train[X_train.index]
y_test=y_test[X_test.index]

X_train['AGE']=X_train['AGE'].fillna(X_train['AGE'].mean())
X_train['CAR_AGE']=X_train['CAR_AGE'].fillna(X_train['CAR_AGE'].mean())
X_train['INCOME']=X_train['INCOME'].fillna(X_train['INCOME'].mean())
X_train['HOME_VAL']=X_train['HOME_VAL'].fillna(X_train['HOME_VAL'].mean())

X_test['AGE']=X_test['AGE'].fillna(X_test['AGE'].mean())
X_test['CAR_AGE']=X_test['CAR_AGE'].fillna(X_test['CAR_AGE'].mean())
X_test['INCOME']=X_test['INCOME'].fillna(X_test['INCOME'].mean())
X_test['HOME_VAL']=X_test['HOME_VAL'].fillna(X_test['HOME_VAL'].mean())


# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
le=LabelEncoder()

for i in range(0,X_train.shape[1]):
    if X_train.dtypes[i]=='object':
        X_train[X_train.columns[i]] = le.fit_transform(X_train[X_train.columns[i]])

for j in range(0,X_test.shape[1]):
    if X_test.dtypes[j]=='object':
        X_test[X_test.columns[j]] = le.fit_transform(X_test[X_test.columns[j]])


# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model=LogisticRegression(random_state=6)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
score=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)


# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote=SMOTE(random_state=6)
X_train,y_train=smote.fit_sample(X_train,y_train)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# Code ends here


# --------------
# Code Starts here
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
score=accuracy_score(y_test,y_pred)
print(score)
# Code ends here


