#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
class rashidijan:

    def __init__(self, D, K):
        self.D = D
        self.K = K
        w1 = torch.randn((D, K), requires_grad=True)
        self.w1 = w1
        b1 = torch.randn((1, K),requires_grad=True)
        self.b1 = b1
        w2 = torch.randn((K, 1),requires_grad=True)
        self.w2 = w2
        b2 = torch.randn((1, 1),requires_grad=True)
        self.b2 = b2
        v1 = torch.randn((D, K),requires_grad=True)
        self.v1 = v1
        c1 = torch.randn((1, K),requires_grad=True)
        self.c1 = c1
        v2 = torch.randn((K, 1),requires_grad=True)
        self.v2 = v2
        c2 = torch.randn((1, 1),requires_grad=True)
        self.c2 = c2

    def mean(self, X):
        self.w1.requires_grad_(True)
        self.w2.requires_grad_(True)
        self.b1.requires_grad_(True)
        self.b2.requires_grad_(True)

        #z = torch.matmul(X, self.w1) + self.b1
        #sigmZ = torch.sigmoid(z)
        #sigmZ = torch.tensor(sigmZ)
        miu = torch.matmul(torch.sigmoid(torch.matmul(X, self.w1) + self.b1), self.w2) + self.b2

        return miu
    def std(self, X):
        self.v1.requires_grad_(True)
        self.v2.requires_grad_(True)
        self.c1.requires_grad_(True)
        self.c2.requires_grad_(True)
        #zz = torch.matmul(X, self.v1) + self.c1
        #sigmZstd = torch.sigmoid(zz)
        #sOmega = torch.matmul(sigmZstd, self.v2) + self.c2
        sigma = torch.exp(torch.matmul(torch.sigmoid(torch.matmul(X, self.v1) + self.c1), self.v2) + self.c2)

        return sigma
    def nll(self, X, y):
        self.w1.requires_grad_(True)
        self.w2.requires_grad_(True)
        self.b1.requires_grad_(True)
        self.b2.requires_grad_(True)
        self.v1.requires_grad_(True)
        self.v2.requires_grad_(True)
        self.c1.requires_grad_(True)
        self.c2.requires_grad_(True)
        NLL = torch.log(self.std(X)) + torch.log(torch.sqrt(torch.tensor(2 * torch.pi))) +              (((y - self.mean(X)) ** 2) / (2 * (self.std(X) ** 2)))
        SNLL = torch.sum(NLL)
        return SNLL
    def fit(self,X,y,alpha,gamma,T):
        alpha = torch.tensor(alpha)
        gamma = torch.tensor(gamma)
        self.w1.requires_grad_(True)
        self.w2.requires_grad_(True)
        self.b1.requires_grad_(True)
        self.b2.requires_grad_(True)
        self.v1.requires_grad_(True)
        self.v2.requires_grad_(True)
        self.c1.requires_grad_(True)
        self.c2.requires_grad_(True)
        pii = 2 * torch.pi
        pii = torch.tensor(pii)
        Output = np.zeros(T)
        outputs = []
        dw10 = torch.zeros((self.D,self.K))
        db10 = torch.zeros((1,self.K))
        dw20 = torch.zeros((self.K,1))
        db20 = torch.zeros((1,1))
        dv10 = torch.zeros((self.D,self.K))
        dv20 = torch.zeros((self.K,1))
        dc10 = torch.zeros((1,self.K))
        dc20 = torch.zeros((1,1))
        dw1 = torch.zeros((self.D,self.K))
        db1 = torch.zeros((1,self.K))
        dw2 = torch.zeros((self.K,1))
        db2 = torch.zeros((1,1))
        dv1 = torch.zeros((self.D,self.K))
        dv2 = torch.zeros((self.K,1))
        dc1 = torch.zeros((1,self.K))
        dc2 = torch.zeros((1,1))
        for t in range(T):
            self.w1.requires_grad_(True)
            self.w2.requires_grad_(True)
            self.b1.requires_grad_(True)
            self.b2.requires_grad_(True)
            self.v1.requires_grad_(True)
            self.v2.requires_grad_(True)
            self.c1.requires_grad_(True)
            self.c2.requires_grad_(True)


            self.nll(X,y).sum().backward()
            with torch.no_grad():

                #dw1.sub_((1 - gamma) * dw10 + alpha * self.w1.grad)
                dw1 = torch.tensor(gamma * dw10 + alpha * self.w1.grad)
                dw2 = torch.tensor(gamma * dw20 + alpha * self.w2.grad)
                db1 = torch.tensor(gamma * db10 + alpha * self.b1.grad)
                db2 = torch.tensor(gamma * db20 + alpha * self.b2.grad)
                dv1 = torch.tensor(gamma * dv10 + alpha * self.v1.grad)
                dv2 = torch.tensor(gamma * dv20 + alpha * self.v2.grad)
                dc1 = torch.tensor(gamma * dc10 + alpha * self.c1.grad)
                dc2 = torch.tensor(gamma * dc20 + alpha * self.c2.grad)

                self.w1 -= dw1
                self.w2 -= dw2
                self.b1 -= db1
                self.b2 -= db2
                self.v1 -= dv1
                self.v2 -= dv2
                self.c1 -= dc1
                self.c2 -= dc2
                dw10 = dw1
                dw20 = dw2
                db10 = db1
                db20 = db2
                dv10 = dv1
                dv20 = dv2
                dc10 = dc1
                dc20 = dc2

            self.w1.grad.zero_()
            self.w2.grad.zero_()
            self.b1.grad.zero_()
            self.b2.grad.zero_()
            self.v1.grad.zero_()
            self.c1.grad.zero_()
            self.v2.grad.zero_()
            self.c2.grad.zero_()



            Output = self.nll(X,y).detach().numpy()
            outputs.append(Output.item())
        W1 = self.w1
        self.W1 = W1
        W2 = self.w2
        self.W2 = W2
        B1 = self.b1
        self.B1 = B1
        B2 = self.b2
        self.B2 = B2
        V1 = self.v1
        self.V1 = V1
        V2 = self.v2
        self.V2 = V2
        C1 = self.c1
        self.C1 = C1
        C2 = self.c2
        self.C2 = C2
        miiu = torch.matmul(torch.sigmoid(torch.matmul(X, self.w1) + self.b1), self.w2) +self.b2
        sigmaaa = torch.exp(torch.matmul(torch.sigmoid(torch.matmul(X, self.v1) + self.c1), self.v2) + self.c2)


        print(self.w1.size())
        #outputs1 = torch.tensor(outputs)

        return outputs, self.w1, self.w2, self.b1, self.b2, self.v1, self.v2, self.c1, self.c2


    def predict(self, X):


        MiuPhi = torch.matmul(torch.sigmoid(torch.matmul(X, self.W1) + self.B1), self.W2) + self.B2
        SigmaOmega = torch.exp(torch.matmul(torch.sigmoid(torch.matmul(X, self.V1) + self.C1), self.V2) + self.C2)
        return MiuPhi, SigmaOmega

