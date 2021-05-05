
'''
Implementation of Supervised Learning using ADALINE
on iris dataset
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

target = []

df = pd.read_csv('iris.csv')
print(df.head())
df1 = df[['Sepal.Length', 'Sepal.Width']]


X= [df1['Sepal.Length'],df1['Sepal.Width']]
X = pd.DataFrame(X).to_numpy()

yy = [df['Species']]
yy = pd.DataFrame(yy).to_numpy()



for i in range(100):
    if yy[0][i] == 'setosa':
        target.append(1)
    else:
        target.append(-1)

w0=0.03
w1=0.03
w2=0.03
error = 0.0
errors=[]
learningRate=0.0001;
epoch = 2000
def step(x):
    if (x > 0):
        return 1
    else:
        return -1;

sum_square_error = []
sum_square_error = [0 for i in range(epoch*100)]
error_sum = 0 
for j in range(epoch):
    for i in range(100):
        y=X[0][i]*w1+X[1][i]*w2+w0;
        pred = step(y)
        
        sum_square_error[i] = sum_square_error[i] + (((target[i] - pred)**2)/(2*100))
        
    w1 = w1 + learningRate*(target[i]-pred)*X[0][i];
    w2 = w2 + learningRate*(target[i]-pred)*X[1][i];
    w0 = w0 + learningRate*(target[i]-pred);
    
print("Updated Weight w0", w0)
print("Updated Weight w1", w1)
print("Updated Weight w2", w2)
print("Learning rate", learningRate)

# testing data after training
a = [5.1,4.9,4.7,6]
b = [3.5,3,3.2,3.4]
for i in range(4):
    if step(w1*a[i]+w2*b[i]+w0) == 1:
        res = 'setosa'
    else:
        res = 'versicolor'
    print("Testing Data", a[i]," " , b[i],"  Results " , res)


ax1 = plt.subplot(111)
ax1.plot(sum_square_error, c='#aaaaff', label='Sum Square Error')
ax1.set_xscale("log")
plt.title("ADALINE Sum Square Errors ")
plt.legend()
plt.xlabel('Error')
plt.ylabel('Value')
plt.show()

