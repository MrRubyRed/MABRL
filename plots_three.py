# -*- coding: utf-8 -*-
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure();

basel_exp = 3;
num_of_reps = 5;
for wexp in range(4):
    #origin = [50,10]#[50,10]#[10,50]#[40,20]#[10,15]
    #h_and_w = [15,70]#[15,70]#[60,20]#[20,50]#[20,42]
    
    #exps = ["[10, 30],[20, 50]","[25, 10],[15, 70]","[42, 10],[20, 42]","[10, 10],[60, 20]"];
    exps = ["[50, 10],[15, 70]","[10, 50],[60, 20]","[40, 20],[20, 50]","[10, 15],[20, 42]"];
    data_1 = [];
    data_2 = [];
    #data_3 = [];    

    for i in range(1,num_of_reps+1):
        data = pickle.load(open("data/plot_two_data/data_"+str(exps[wexp])+"up3_"+str(i)+".pkl", "rb" ))
        data_1.append(np.array(data[0][:-1]));
        t = np.array([t for t in range(1,len(data_1[-1])+1)]);
                     
        data_1[-1] = data_1[-1]/t
        data_1[-1] = data_1[-1][None]
    
    for i in range(basel_exp):
        data_2.append(np.array(pickle.load(open("data/plot_three_data/experts_100avg_performance_exp"+str(i+4)+"_"+str(wexp+1)+"_.pkl", "rb" ))))
    
    data_3 = np.array(pickle.load(open("data/plot_three_data/experts_100avg_performance_exp_b_"+str(wexp+1)+"_.pkl", "rb" )))    
    data_4 = np.array(pickle.load(open("data/plot_three_data/experts_100avg_performance_exp_b2_"+str(wexp+1)+"_.pkl", "rb" )))    
    data_5 = np.array(pickle.load(open("data/plot_three_data/experts_100avg_performance_exp_b3_"+str(wexp+1)+"_.pkl", "rb" )))    
    
    hund_step_rew = np.concatenate(data_1,axis=0)
    data_2 = np.concatenate(data_2,axis=0)
    data_3 = np.concatenate((data_3,data_4,data_5),axis=0)
    
    ax = plt.subplot(2,2,wexp+1)
    sns.tsplot(data=hund_step_rew, color='k');
    sns.tsplot(data=data_2,condition=["Random O."], color='b')
    sns.tsplot(data=data_3,condition=["No O. "], color='g')
    #sns.tsplot(data=fixed_cum_rew, condition=["Fixed Exp. "+str(wexp+1)+" (100-S Avg. Rew)"], legend=True, color='b');       
    plt.xlim(0,1500); ax.set_xticks(ax.get_xticks()[::2]); plt.xlim(0,1500);
    plt.ylim(0,0.08); ax.set_yticks(ax.get_yticks()[::2]); plt.ylim(0,0.08); 
    #plt.title("100-Step Reward");
    #plt.xlabel("Bandit Timestep");
    #plt.ylabel("100-Step Cum. Reward/100N");           

fig.text(0.5, 0.04, 'Iteration Step', ha='center', va='center')
fig.text(0.03, 0.5, 'Iteration-Normalized', ha='center', va='center', rotation='vertical')
fig.text(0.06, 0.5, 'Cumulative Reward', ha='center', va='center', rotation='vertical')
               
plt.show();