#Linear Classifier from scratch
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


#分類
def predict(result):
    if result > 0:
        return 1
    else:
        return -1
fig_len=data_dum.shape[1]-1
id_len=data_dum.shape[0]

#設定weight
w=np.zeros(fig_len+1)   

#計算錯誤次數
def error_num(wi):
    empty=[]
    wn=np.append(empty,wi)
    
    pre=[]
    for i1 in range(id_len):
            x1=np.concatenate((np.array([1]),np.array(data_dum.iloc[i1])[1:]))
            p=predict(np.dot(wn,x1))
            pre.append(p)

    b=0
    for i2 in range(id_len):
        y1=np.array(data_dum.iloc[i2])[0]
        if pre[i2]!=y1 :
            b=b+1
    return b



error = 1
iterator = 0
failure=[id_len]
while error !=0:
    error=0
    for i in range(id_len):
        x=np.concatenate((np.array([1]),np.array(data_dum.iloc[i])[1:]))
        y=np.array(data_dum.iloc[i])[0]
        if predict(np.dot(w,x)) != y:
            iterator+=1
            error+=1
            w=w+y*x
            xi=[]
            xi.append(x)
            wi=[]
            wi.append(w)
            it=[]
            it.append(iterator)
            error_count=error_num(wi)
            if error_count < min(failure) :
                final_wi=[]
                final_wi=w
                min_error=[]
                min_error=error_count
            failure.append(error_count)

        if  iterator>500 :
            break     

print("wi:",final_wi)
print("Accuracy"+str((id_len-min_error)/id_len))
