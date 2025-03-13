#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import matplotlib.pyplot as plt
from nn_base import hsrnn 
from nn_mods import hsrnnMode 
data = np.load("../data/data.npz")
X_train = torch.tensor(data['X_train']).float()
X_test = torch.tensor(data["X_test"]).float()
y_train = torch.tensor(data["y_train"]).float()
y_test = torch.tensor(data["y_test"]).float()
i = X_train.detach().numpy()
i=np.shape(i)
rt = hsrnn(i[1],3)

N=5
T=50000
train = np.zeros((N, T))
test = np.zeros((N, T))
ribbon = np.zeros((N*2, 201))
Miu = np.zeros((N, 201))
mm = np.zeros((i[0], N))
Sigma = np.zeros(N)
SigmA = np.zeros((N, 201))
a = np.array(range(0,201))
aa = (a/100)
z = np.reshape(aa, (201,1))
x = torch.tensor(z).float()
X = np.reshape(z, ((201,)))
trainNLL = torch.zeros(1, N)
testNLL = torch.zeros(1, N)

for n in range(0,5):
    rt_n = hsrnn(i[1],3)
    train_n = rt_n.fit(X_train, y_train, 0.0001, 0.9, T)[0]
    MiuTest_n = rt_n.predict(X_test)[0]
    SigmaTest_n = rt_n.predict(X_test)[1]
    testnll_n = torch.sum(torch.log(SigmaTest_n) + torch.log(torch.sqrt(torch.tensor(2 * torch.pi))) +              (((y_test - MiuTest_n) ** 2) / (2 * (SigmaTest_n ** 2))))
    ribMiu_n = rt_n.predict(x)[0]
    Miuu_n = ribMiu_n.detach().numpy()

    ribSig_n = rt_n.predict(x)[1]
    Sigmaa_n = ribSig_n.detach().numpy()

    ValueMiu = np.reshape(Miuu_n, (1,201))
    Miu[n,:] = ValueMiu

    ValueSigma = np.reshape(Sigmaa_n, (1,201))
    SigmA[n,:] = ValueSigma

    train[n,] = train_n
    trainNLL[0,n] = train_n[-1]
    testNLL[0,n] = testnll_n


print(trainNLL)
print(testNLL)

plt.plot(range(T), train[0,], color='r', label='set 1')
plt.plot(range(T), train[1,], color='blue', label='set 2')
plt.plot(range(T), train[2,], color='orange', label='set 3')
plt.plot(range(T), train[3,], color='g', label='theta4')
plt.plot(range(T), train[4,], color='black', label='theta5')
plt.ylim(0,700)
plt.xlabel('epochs')
plt.ylabel('NLL')
plt.title('Question 6')
plt.legend()
plt.show()
print(np.shape(Miu))
print(np.shape(SigmA))
adding = Miu + SigmA
Adding = np.reshape(adding, (201,5))
subtrac = Miu - SigmA
Subtrac = np.reshape(subtrac, (201,5))

#********
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='b', label = 'Train_data')
ax.fill_between(X, subtrac[0,:], adding[0,:], label = 'Miu -+ 2*sigma')
plt.scatter(X_train, y_train, color='b',alpha = 0.2)
plt.xlim(0,2.00)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Question 6-c')
plt.legend()
plt.show()
#********
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='b', label = 'Train_data')
ax.fill_between(X, subtrac[1,:], adding[1,:], label = 'Miu -+ 2*sigma')
plt.scatter(X_train, y_train, color='b',alpha = 0.2)
plt.xlim(0,2.00)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Question 6-c')
plt.legend()
plt.show()
#********
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='b', label = 'Train_data')
ax.fill_between(X, subtrac[2,:], adding[2,:])
plt.scatter(X_train, y_train, color='b',alpha = 0.2)
plt.xlim(0,2.00)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Question 6-c')
plt.legend()
plt.show()
#*********
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='b', label = 'Train_data')
ax.fill_between(X, subtrac[3,:], adding[3,:])
plt.scatter(X_train, y_train, color='b',alpha = 0.2)
plt.xlim(0,2.00)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Question 6-c')
plt.legend()
plt.show()
#**********
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='b', label = 'Train_data')
ax.fill_between(X, subtrac[4,:], adding[4,:])
plt.scatter(X_train, y_train, color='b',alpha = 0.2)
plt.xlim(0,2.00)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Question 6-c')
plt.legend()
plt.show()
#************
#6d



Nn=11
T=50000


tRain = np.zeros((Nn, T))
tEst = np.zeros((Nn, T))
ribbon = np.zeros((Nn*2, 201))
MMIIUU = np.zeros((Nn, 201))
mmm = np.zeros((i[0], Nn))
SigMA = np.zeros(Nn)
SigMAA = np.zeros((Nn, 201))
a = np.array(range(0,201))
aa = (a/100)
z = np.reshape(aa, (201,1))
x = torch.tensor(z).float()
X = np.reshape(z, ((201,)))
tRainNLL = torch.zeros(1, Nn)
tEstNLL = torch.zeros(1, Nn)

for m in range(0,11):
    rx_m = hsrnn(i[1],m)
    tRain_m = rx_m.fit(X_train, y_train, 0.0001, 0.9, T)[0]
    MMIIUUTest_m = rx_m.predict(X_test)[0]
    SigMATest_m = rx_m.predict(X_test)[1]
    tEstnll_m = torch.sum(torch.log(SigMATest_m) + torch.log(torch.sqrt(torch.tensor(2 * torch.pi))) +              (((y_test - MMIIUUTest_m) ** 2) / (2 * (SigMATest_m ** 2))))
    ribMMIIUU_m = rx_m.predict(x)[0]
    MMIIUUu_m = ribMMIIUU_m.detach().numpy()

    ribSig_m = rx_m.predict(x)[1]
    SigMAa_m = ribSig_m.detach().numpy()

    ValueMMIIUU = np.reshape(MMIIUUu_m, (1,201))
    MMIIUU[m,:] = ValueMMIIUU

    ValueSigMA = np.reshape(SigMAa_m, (1,201))
    SigMAA[m,:] = ValueSigMA

    tRain[m,] = tRain_m
    tRainNLL[0,m] = tRain_m[-1]
    tEstNLL[0,m] = tEstnll_m


print(tRainNLL)
print(tEstNLL)
plt.plot(range(T), tRain[1,], 'y--', label='K = 0')
plt.plot(range(T), tRain[1,], 'r--', label='K = 1')
plt.plot(range(T), tRain[2,], color='blue', label='K = 2')
plt.plot(range(T), tRain[3,], color='orange', label='K = 3')
plt.plot(range(T), tRain[4,], color='g', label='K = 4')
plt.plot(range(T), tRain[5,], color='black', label='K = 5')
plt.plot(range(T), tRain[6,], color='brown', label='K = 6')
plt.plot(range(T), tRain[7,], color='y', label='K = 7')
plt.plot(range(T), tRain[8,], color='r', label='K = 8')
plt.plot(range(T), tRain[9,], 'k', label='K = 9 ')
plt.plot(range(T), tRain[10,], 'c--', label='K = 10')
plt.ylim(0,700)
plt.xlabel('epochs')
plt.ylabel('NLL')
plt.title('Question 6')
plt.legend()
plt.show()
print(np.shape(MMIIUU))
print(np.shape(SigMAA))
ADDING = MMIIUU + SigMAA
aDDING = np.reshape(ADDING, (201,11))
SUBtrac = MMIIUU - SigMAA
sUBTRAC = np.reshape(SUBtrac, (201,11))

#********
#**********

fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='b', label = 'Train_data')
ax.fill_between(X, SUBtrac[0,:], ADDING[0,:], label = 'MMIIUU -+ 2*sigma',alpha = 0.2)
plt.xlim(0,2.00)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Question 6-d')
plt.legend()
plt.show()
#********
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='b', label = 'Train_data')
ax.fill_between(X, SUBtrac[1,:], ADDING[1,:], label = 'MMIIUU -+ 2*sigma',alpha = 0.2)
plt.xlim(0,2.00)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Question 6-d')
plt.legend()
plt.show()
#********
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='b', label = 'Train_data')
ax.fill_between(X, SUBtrac[2,:], ADDING[2,:], label = 'MMIIUU -+ 2*sigma',alpha = 0.2)
plt.xlim(0,2.00)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Question 6-d')
plt.legend()
plt.show()
#*********
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='b', label = 'Train_data')
ax.fill_between(X, SUBtrac[3,:], ADDING[3,:], label = 'MMIIUU -+ 2*sigma',alpha = 0.2)
plt.xlim(0,2.00)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Question 6-d')
plt.legend()
plt.show()
#**********
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='b', label = 'Train_data')
ax.fill_between(X, SUBtrac[4,:], ADDING[4,:], label = 'MMIIUU -+ 2*sigma',alpha = 0.2)
plt.xlim(0,2.00)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Question 6-d')
plt.legend()
plt.show()
#**********
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='b', label = 'Train_data')
ax.fill_between(X, SUBtrac[5,:], ADDING[5,:], label = 'MMIIUU -+ 2*sigma',alpha = 0.2)
plt.xlim(0,2.00)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Question 6-d')
plt.legend()
plt.show()
#**********
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='b', label = 'Train_data')
ax.fill_between(X, SUBtrac[6,:], ADDING[6,:], label = 'MMIIUU -+ 2*sigma',alpha = 0.2)
plt.xlim(0,2.00)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Question 6-d')
plt.legend()
plt.show()
#**********
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='b', label = 'Train_data')
ax.fill_between(X, SUBtrac[7,:], ADDING[7,:], label = 'MMIIUU -+ 2*sigma',alpha = 0.2)
plt.xlim(0,2.00)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Question 6-d')
plt.legend()
plt.show()
#**********
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='b', label = 'Train_data')
ax.fill_between(X, SUBtrac[8,:], ADDING[8,:], label = 'MMIIUU -+ 2*sigma',alpha = 0.2)
plt.xlim(0,2.00)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Question 6-d')
plt.legend()
plt.show()
#**********
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='b', label = 'Train_data')
ax.fill_between(X, SUBtrac[9,:], ADDING[9,:], label = 'MMIIUU -+ 2*sigma',alpha = 0.2)
plt.xlim(0,2.00)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Question 6-d')
plt.legend()
plt.show()
#**********
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='b', label = 'Train_data')
ax.fill_between(X, SUBtrac[10,:], ADDING[10,:], label = 'MMIIUU -+ 2*sigma',alpha = 0.2)
plt.xlim(0,2.00)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Question 6-d')
plt.legend()
plt.show()

#***********
#6e
#different Gammas
trainG = np.zeros((N, T))
testG = np.zeros((N, T))
trainNLLG= torch.zeros(1, N)
testNLLG = torch.zeros(1, N)
Gamma = [0.1,0.3, 0.5,0.7, 0.9]
for n in range(0,5):
    ModsG_n = hsrnnMode(1,3)
    trainG_n = ModsG_n.fit(X_train, y_train, 0.0001, Gamma[n], T)[0]
    MiuTestG_n = ModsG_n.predict(X_test)[0]
    SigmaTestG_n = ModsG_n.predict(X_test)[1]
    testnllG_n = torch.sum(torch.log(SigmaTestG_n) + torch.log(torch.sqrt(torch.tensor(2 * torch.pi))) +                           (((y_test - MiuTestG_n) ** 2) / (2 * (SigmaTestG_n ** 2))))


    trainG[n,] = trainG_n
    trainNLLG[0, n] = trainG_n[-1]
    testNLLG[0, n] = testnllG_n

print(trainNLLG)
print(testNLLG)

plt.plot(range(T), trainG[0,], color='r', label='Gamma = 0.1')
plt.plot(range(T), trainG[1,], color='blue', label='Gamma = 0.3')
plt.plot(range(T), trainG[2,], color='orange', label='Gamma = 0.5')
plt.plot(range(T), trainG[3,], color='g', label='Gamma = 0.7')
plt.plot(range(T), trainG[4,], color='black', label='Gamma = 0.9')
plt.ylim(0,700)
plt.xlabel('epochs')
plt.ylabel('NLL')
plt.title('Question 6 - e - different Gammas in GD')
plt.legend()
plt.show()
# different Epochs
trainNLLT= torch.zeros(1, N)
testNLLT = torch.zeros(1, N)
epoch = [100,1000,10000, 20000,T]
trainnt = torch.zeros((1,T))
for n in range(0,5):
    ModsT_n = hsrnnMode(1,3)
    trainT_n = ModsT_n.fit(X_train, y_train, 0.0001, 0.9, epoch[n])[0]
    MiuTestT_n = ModsT_n.predict(X_test)[0]
    SigmaTestT_n = ModsT_n.predict(X_test)[1]
    testnllT_n = torch.sum(torch.log(SigmaTestT_n) + torch.log(torch.sqrt(torch.tensor(2 * torch.pi))) +                           (((y_test - MiuTestT_n) ** 2) / (2 * (SigmaTestT_n ** 2))))
    if n == 4:
        trainnt = trainT_n
    trainnt


    trainNLLT[0, n] = trainT_n[-1]
    testNLLT[0, n] = testnllT_n


print(trainNLLT)
print(testNLLT)

plt.plot(range(T), trainnt, label='T=50000')
plt.plot(epoch[0],trainnt[epoch[0]-1], 'r*', label = 'T=100' )
plt.plot(epoch[1],trainnt[epoch[1]-1], 'g*', label = 'T=1000' )
plt.plot(epoch[2],trainnt[epoch[2]-1], 'k^', label = 'T=10000' )
plt.plot(epoch[3],trainnt[epoch[3]-1], 'm^', label = 'T=20000' )
plt.plot(epoch[4],trainnt[epoch[4]-1], 'ys', label = 'T=50000' )
plt.ylim(0,700)
plt.xlabel('epochs')
plt.ylabel('NLL')
plt.title('Question 6 - e - different Epochs in GD')
plt.legend()
plt.show()

#**********
#Nesterov

ModsT = hsrnnMode(1,3)

NNN=1
T=50000

trainNLLNes = torch.zeros(1, 1)
testNLLNes = torch.zeros(1, 1)

trainntNes = torch.zeros((1,T))


#******

ModsTNes = hsrnnMode(1,3)
trainNes = ModsTNes.fitNesterov(X_train, y_train, 0.0001, 0.9, T)[0]
MiuTestNes = ModsTNes.predict(X_test)[0]
SigmaTestTNes = ModsTNes.predict(X_test)[1]
testNes = torch.sum(torch.log(SigmaTestTNes) + torch.log(torch.sqrt(torch.tensor(2 * torch.pi))) +
                       (((y_test - MiuTestNes) ** 2) / (2 * (SigmaTestTNes ** 2))))

trainNLLNes[0, 0] = trainNes[-1]
testNLLNes[0, 0] = testNes

#*****
trainNLLNoNes = torch.zeros(1, 1)
testNLLNoNes = torch.zeros(1, 1)

trainntNoNes = torch.zeros((1,T))


ModsNoTNes = hsrnnMode(1,3)
trainNoNes = ModsNoTNes.fit(X_train, y_train, 0.0001, 0.9, T)[0]
MiuTestNoNes = ModsNoTNes.predict(X_test)[0]
SigmaTestNoTNes = ModsNoTNes.predict(X_test)[1]
testNoNes = torch.sum(torch.log(SigmaTestNoTNes) + torch.log(torch.sqrt(torch.tensor(2 * torch.pi))) +
                       (((y_test - MiuTestNoNes) ** 2) / (2 * (SigmaTestNoTNes ** 2))))

trainNLLNoNes[0, 0] = trainNoNes[-1]
testNLLNoNes[0, 0] = testNoNes

print(trainNoNes)
print(trainNes)
plt.plot(range(T), trainNoNes,'c' ,label='GD')
plt.plot(range(T), trainNes,'k--' ,label='Nesterov GD')
plt.ylim(0,700)
plt.xlabel('epochs')
plt.ylabel('NLL')
plt.title('Question 6 - e - Comparing GD with Neterov GD')
plt.legend()
plt.show()


