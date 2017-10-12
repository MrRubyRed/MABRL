# -*- coding: utf-8 -*-
import numpy as np
import pickle
import matplotlib.pyplot as plt

N = 4;

#["[10, 30]_[20, 50]","[25, 10]_[15, 70]","[42, 10]_[20, 42]","[10, 10]_[60, 20]"];
data = pickle.load(open("data/plot_two_data/data_[28, 28],[28, 28]up3_1.pkl", "rb" ));
#r_data = pickle.load(open("data/experts_100avg_performance_exp1.pkl", "rb" ));
x = np.linspace(1,len(data[0]),num=len(data[0]));
y = np.concatenate(data[1][:],axis=0)
t = np.array([t for t in range(1,len(x))]);
    
plt.subplot(2,1,1)
plt.plot(data[0][:-1]/t);
#plt.plot([r_data[0][-1]]*(len(x)-1));
plt.xlim(0,1500);
plt.subplot(2,1,2)                   
for i in range(N):
    plt.plot(y[:,i]/t);
plt.xlim(0,1500);        