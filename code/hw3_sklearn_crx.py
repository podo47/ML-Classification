#SVM by sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

#匯入csv檔
dt = pd.read_csv('crx.csv')
dt1=pd.DataFrame(dt)
#去除含missing value欄位
col_len=dt1.shape[1]
row_len=dt1.shape[0]
for i in range(row_len):
    for p in range(col_len):
        if dt1.iloc[i,p]=='?' :
            dt1.iloc[i,p]=np.nan
data = dt1.dropna()
#修改label欄位為1和-1
col_len2=data.shape[1]
row_len2=data.shape[0]
for i in range(row_len2):
    if data.iloc[i,col_len2-1]=='+' :
        data.iloc[i,col_len2-1]=1
    if data.iloc[i,col_len2-1]=='-' :
        data.iloc[i,col_len2-1]=-1 

#onehot-encoding
data[['att2', 'att3','att8','att11','att14','att15','label']]=data[['att2', 'att3','att8','att11','att14','att15','label']].astype(float)
data_dum = pd.get_dummies(data)
data_label=data_dum['label']
data_dum=data_dum.drop('label',1)
data_dum=pd.concat([data_label,data_dum],axis=1)
print(data_dum)

data_x=np.array(data_dum.iloc[:,1:])
data_y=np.array(data_dum.iloc[:,0])

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0)
sklearn_svm_linear = svm.SVC(kernel='linear') # Linear Kernel
sklearn_svm_linear.fit(X_train, y_train)
y_pred = sklearn_svm_linear.predict(X_test)
print("SVM_linear Accuracy:",metrics.accuracy_score(y_test, y_pred))
