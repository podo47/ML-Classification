#LC with least-squared manner
import numpy as np
import pandas as pd

#匯入csv檔
dt = pd.read_csv('crx.csv')
dt1=pd.DataFrame(dt)
#去除missing value
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
#data shape
fig_len=data_dum.shape[1]-1
id_len=data_dum.shape[0]

#建立 x 和 y 的 array
x0=np.concatenate((np.array(data_dum.iloc[0])[1:],np.array([1])))
for i in range(1,id_len):
    x=np.concatenate((np.array(data_dum.iloc[i])[1:],np.array([1])))
    if i==1 :
        x1=np.vstack((x0,x))
    else:
        x1=np.vstack((x1,x))

y0=np.array(data_dum.iloc[0])[0]
for i1 in range(1,id_len):
    y=np.array(data_dum.iloc[i1])[0]
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
#Accracy: 0.5344563552833078
