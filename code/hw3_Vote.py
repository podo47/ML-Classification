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

#分類
def predict(result):
    if result > 0:
        return 1
    else:
        return -1

#計算正確次數
def right_num(wi):
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
        if pre[i2]==y1 :
            b=b+1
    return b

class VotedPerceptron:
    def __init__(self, n_iter):
        self.n_iter = n_iter
        self.W = []
        self.true = []
        self.y_pred=[]

    def train(self):
        #設定weight
        w=np.zeros(fig_len+1)  
        for epoch in range(self.n_iter):
            for i in range(id_len):
                x=np.concatenate((np.array([1]),np.array(data.iloc[i])[1:]))
                y=np.array(data.iloc[i])[0]
                if predict(np.dot(w,x)) != y:
                    self.W.append(w)
                    correct=right_num(w)
                    self.true.append(correct)
                    w=w+y*x
                if right_num(w)==id_len:
                    print("Find best w:",w)

    def test(self):
        for i3 in range(id_len):
            pred=[]
            x2=np.concatenate((np.array([1]),np.array(data.iloc[i3])[1:]))
            for h in range(len(self.W)):
                pr=predict(np.dot(self.W[h],x2))*self.true[h]
                pred.append(pr)
            if sum(pred) < 0:
                self.y_pred.append(-1)
            else:
                self.y_pred.append(1)

    def error(self):
        er=0
        for i4 in range(id_len):
            y2=np.array(data.iloc[i4])[0]
            if self.y_pred[i4]!=y2:
                er+=1
        print("Accracy=",(id_len-er)/id_len)


data1=VotedPerceptron(n_iter=20)
data1.train()
data1.test()
data1.error()