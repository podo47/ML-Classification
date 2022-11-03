#LC with least-squared manner
import numpy as np
import pandas as pd

#匯入csv檔
dt = pd.read_csv('data.csv')
dt1=pd.DataFrame(dt)
data = dt1.drop(["ID"], axis=1)

#修改Diagnosis欄位為1和-1
Diagnosis={'M':1,'B':-1}
data['Diagnosis']=data['Diagnosis'].map(Diagnosis)

#data shape
fig_len=data.shape[1]-1  #data行數
id_len=data.shape[0]  #data列數 

#建立 x 和 y 的 array
x0=np.concatenate((np.array(data.iloc[0])[1:],np.array([1])))
for i in range(1,id_len):
    x=np.concatenate((np.array(data.iloc[i])[1:],np.array([1])))
    if i==1 :
        x1=np.vstack((x0,x))
    else:
        x1=np.vstack((x1,x))

y0=np.array(data.iloc[0])[0]
for i1 in range(1,id_len):
    y=np.array(data.iloc[i1])[0]
    if i1==1 :
        y1=np.vstack((y0,y))
    else:
        y1=np.vstack((y1,y))

#Least-squares solution method
xt=x1.T
n_dot=np.dot(xt,x1)
inverse=np.linalg.inv(n_dot)
w0=np.dot(inverse,xt)
w=np.dot(w0,y1)


predict_y=np.dot(x1,w)


for i2 in range(id_len):
    if predict_y[i2]>=0:
        predict_y[i2]=1
    else:
        predict_y[i2]=-1

#計算預測錯誤次數
error=0
for i3 in range(id_len):
    if predict_y[i3] != y1[i3]:
        error=error+1

print('Accracy:',(id_len-error)/id_len) 
#Accracy: 0.9648506151142355


