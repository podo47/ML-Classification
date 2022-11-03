import pandas as pd
import numpy as np
import cvxopt
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

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

#data shape
fig_len=data_dum.shape[1]-1
id_len=data_dum.shape[0]

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

class SVM(object):

    def __init__(self,C, kernel=linear_kernel):
        self.kernel = kernel
        self.C = C

    def fit(self,X,y):
        n_samples=id_len
        n_features =fig_len
        self.y=y
        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        Q = cvxopt.matrix(np.outer(y,y) * K)
        p = cvxopt.matrix(-np.ones(n_samples))
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * self.C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(Q, p, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None
        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)
    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b
    def predict(self, X):
        pred=np.sign(self.project(X))
        return pred

def test_soft(C):
        x=np.array(data_dum.iloc[:,1:])
        y=np.array(data_dum.iloc[:,0])
        y=y.astype('float')
        soft=SVM(C)
        soft.fit(x,y)
        soft.project(x)
        predict_y = soft.predict(x)
        correct = np.sum(predict_y == y)
        acc=correct/len(predict_y)
        return acc   

def draw_different_C(start,end):
    value_c=[]
    index=list(range(start,end+1))
    for c in index:
        c1=test_soft(c)
        value_c.append(c1)
    plt.plot(index,value_c)
    plt.title("crx-Performance with different C") # title
    plt.ylabel("Accuracy") # y label
    plt.xlabel("C") # x label
    plt.grid(True)
    plt.savefig('Performance with different C_crx')
    best_C=max(value_c)
    print("Best accuracy=%f" % best_C)

draw_different_C(1,10)
#Best accuracy=0.875957
