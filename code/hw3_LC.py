import numpy as np
import pandas as pd

#匯入csv檔
dt = pd.read_csv('data.csv')
dt1=pd.DataFrame(dt)
data = dt1.drop(["ID"], axis=1)

#修改Diagnosis欄位為1和-1
Diagnosis={'M':1,'B':-1}
data['Diagnosis']=data['Diagnosis'].map(Diagnosis)

#分類
def predict(result):
    if result > 0:
        return 1
    else:
        return -1
fig_len=data.shape[1]-1
id_len=data.shape[0]
print(id_len)

#設定weight
w=np.zeros(fig_len+1)   

#計算錯誤次數
def error_num(wi):
    empty=[]
    wn=np.append(empty,wi)
    
    pre=[]
    for i1 in range(id_len):
            x1=np.concatenate((np.array([1]),np.array(data.iloc[i1])[1:]))
            p=predict(np.dot(wn,x1))
            pre.append(p)

    b=0
    for i2 in range(id_len):
        y1=np.array(data.iloc[i2])[0]
        if pre[i2]!=y1 :
            b=b+1
    return b



error = 1
iterator = 0
failure=[569]
while error !=0:
    error=0
    for i in range(id_len):
        x=np.concatenate((np.array([1]),np.array(data.iloc[i])[1:]))
        y=np.array(data.iloc[i])[0]
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

print("wi:"+str(final_wi))
print("Accuracy"+str((id_len-min_error)/id_len))
