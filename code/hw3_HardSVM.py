#Hard margin SVM
import pandas as pd
import numpy as np
import cvxopt
from cvxopt import matrix, solvers

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

def test(w):
    pred=[]
    for i2 in range(id_len):
        x2=np.concatenate((np.array([1]),np.array(data.iloc[i2])[1:]))
        pr=predict(np.dot(w,x2))
        pred.append(pr)
    return pred


def error(prediction):
    er=0
    for i3 in range(id_len):
        y2=np.array(data.iloc[i3])[0]
        if prediction[i3]!=y2:
            er+=1
    print("Accracy=",(id_len-er)/id_len)

s_train=SVM_train(data)
s_test=test(s_train)
error(s_test)
#Accracy= 1.0

#作業第一題的linear classifier 的 w值
LinearClassifier_w=[-1.1100000e+02, -8.5071400e+02, -1.4505000e+03, -5.1172600e+03,
 -4.0078000e+03, -9.0991100e+00, -8.7776000e-01,  9.0879033e+00,
  4.1028410e+00, -1.7260500e+01, -6.7012500e+00, -6.9538000e+00,
 -1.0892500e+02, -2.7965400e+01,  1.4809200e+03, -6.7532700e-01,
 -2.8281900e-01,  2.2655130e-01, -2.0084200e-01, -1.9025510e+00,
 -2.5584190e-01, -8.6832400e+02, -1.9496200e+03, -5.1309600e+03,
  4.5168000e+03, -1.1988050e+01,  2.6181900e+00,  1.6254263e+01,
  3.4314490e+00, -2.5557900e+01, -7.2516200e+00]

margin2_len=1/np.linalg.norm(LinearClassifier_w)
print("Conventional linear classifier Margin = ",margin2_len)
#Conventional linear classifier Margin = 0.00010071789967337403
