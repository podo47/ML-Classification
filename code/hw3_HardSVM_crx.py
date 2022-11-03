import numpy as np
import pandas as pd
import cvxopt
from cvxopt import matrix, solvers

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

def SVM_train(data):
    n_fig_len=fig_len+1
    x=np.array(data.iloc[:,1:])
    new_x=np.ones((id_len,n_fig_len))
    y=np.array(data.iloc[:,0])
    new_x[:,1:]=x
    new_xx=np.ones((id_len,n_fig_len))
    for i in range(0,n_fig_len):
        new_xx[:,i]=y*new_x[:,i]
    # 建立Quadratic programming的Q矩陣
    new_Q=np.diag(np.ones(fig_len+1))
    new_Q[0,0]=0
    Q=cvxopt.matrix(new_Q)
    # 建立Quadratic programming的c矩陣
    p = cvxopt.matrix(np.zeros(n_fig_len))
    # 建立Quadratic programming的A矩陣
    G = cvxopt.matrix(-new_xx)
    # 建立Quadratic programming的b矩陣
    h = cvxopt.matrix(-np.ones(id_len))
    # 利用cvxopt套件求解
    solution = cvxopt.solvers.qp(Q, p, G, h)
    wb=np.array(solution['x'])
    wi=wb[1:].flatten()
    wb_final=wb.flatten()
    margin_len=1/np.linalg.norm(wi)
    print("Hard SVM Margin = ",margin_len)
    return wb_final

#分類
def predict(result):
    if result > 0:
        return 1
    else:
        return -1

def test(w,data):
    pred=[]
    for i2 in range(id_len):
        x2=np.concatenate((np.array([1]),np.array(data.iloc[i2])[1:]))
        pr=predict(np.dot(w,x2))
        pred.append(pr)
    return pred


def error(prediction,data):
    er=0
    for i3 in range(id_len):
        y2=np.array(data.iloc[i3])[0]
        if prediction[i3]!=y2:
            er+=1
    print("Accracy=",(id_len-er)/id_len)

s_train=SVM_train(data_dum)
s_test=test(s_train,data_dum)
error(s_test,data_dum)

#作業第一題的linear classifier 的 w值
LinearClassifier_w=[-3.,7.33,-49.88,55.21, 53., -392.,419., -7., 4.,
1.,2.,-6., 2., 1.,-6.,-1.,-4., 2.,0.,-2., -2.,-4., -1.,1.,4., 1.,  0.,
4.,-1.,-2.,  0.,-1., -2., 0.,-1.,-1.,4.,0.,-14.,11.,-9., 6., -6.,  3., -7.,0., 4.]

margin2_len=1/np.linalg.norm(LinearClassifier_w)
print("Conventional linear classifier Margin = ",margin2_len)
