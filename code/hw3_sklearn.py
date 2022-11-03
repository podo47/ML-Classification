#SVM by sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

#匯入csv檔
dt = pd.read_csv('data.csv')
dt1=pd.DataFrame(dt)
data = dt1.drop(["ID"], axis=1)

#修改Diagnosis欄位為1和-1
Diagnosis={'M':1,'B':-1}
data['Diagnosis']=data['Diagnosis'].map(Diagnosis)

data_x=np.array(data.iloc[:,1:])
data_y=np.array(data.iloc[:,0])

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0)
sklearn_svm_linear = svm.SVC(kernel='linear') # Linear Kernel
sklearn_svm_linear.fit(X_train, y_train)
y_pred = sklearn_svm_linear.predict(X_test)
print("SVM_linear Accuracy:",metrics.accuracy_score(y_test, y_pred))
#SVM_linear Accuracy: 0.9590643274853801
