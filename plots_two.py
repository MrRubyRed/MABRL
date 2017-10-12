# -*- coding: utf-8 -*-
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure();

N = 4;
num_of_reps = 5;
for wexp in range(4):
    #origin = [50,10]#[50,10]#[10,50]#[40,20]#[10,15]
    #h_and_w = [15,70]#[15,70]#[60,20]#[20,50]#[20,42]
    
    #exps = ["[10, 30],[20, 50]","[25, 10],[15, 70]","[42, 10],[20, 42]","[10, 10],[60, 20]"];
    exps = ["[50, 10],[15, 70]","[10, 50],[60, 20]","[40, 20],[20, 50]","[10, 15],[20, 42]"];
    data_1 = [];
    data_2 = [];
    r_data = [];
    
    for i in range(1,num_of_reps+1):
        data = pickle.load(open("data/plot_two_data/data_"+str(exps[wexp])+"up3_"+str(i)+".pkl", "rb" ))
        #print(str(data[2]))
        data_1.append(np.array(data[0][:-1]));
        #data_1[-1] = data_1[-1][None][None].T
        t = np.array([t for t in range(1,len(data_1[-1])+1)]);
                     
        data_1[-1] = data_1[-1]/t
        data_1[-1] = data_1[-1][None]
    
        tmp = np.concatenate(data[1][:],axis=0)/t[None].T
        data_2.append(tmp[None].T)
    
    for i in range(N):
        r_data.append(pickle.load(open("data/plot_two_data/experts_100avg_performance_exp_"+str(i)+"_"+str(wexp+1)+"_.pkl", "rb" )))
        r_data[-1] = np.array([np.mean(r_data[-1])]*len(r_data[-1][0]))
    #r_data = np.mean(pickle.load(open("data/plot_one_data/experts_100avg_performance_exp"+str(wexp)+".pkl", "rb" )))
    #r_data = np.concatenate([np.array([r_data])]*1500)[None];
    
    #x = np.linspace(1,len(data[0]),num=len(data[0]));
    hund_step_rew = np.concatenate(data_1,axis=0)
    #fixed_cum_rew = np.concatenate(r_data[None],axis=0)
    
    picks = np.concatenate(data_2,axis=2).T
    ax = plt.subplot(4,2,2*wexp+1)
    sns.tsplot(data=r_data[0], color='r', linestyle='--');
    sns.tsplot(data=r_data[1], color='g', linestyle='--');
    sns.tsplot(data=r_data[2], color='b', linestyle='--');
    sns.tsplot(data=r_data[3], color='c', linestyle='--');
    sns.tsplot(data=hund_step_rew, color='k');
    #sns.tsplot(data=fixed_cum_rew, condition=["Fixed Exp. "+str(wexp+1)+" (100-S Avg. Rew)"], legend=True, color='b');       
    plt.xlim(0,1500); ax.set_xticks(ax.get_xticks()[::2]); plt.xlim(0,1500); 
    plt.ylim(0,0.08); ax.set_yticks(ax.get_yticks()[::2]); plt.ylim(0,0.08);
    #plt.title("100-Step Reward");
    #plt.xlabel("Bandit Timestep");
    #plt.ylabel("100-Step Cum. Reward/100N");
    
    ax = plt.subplot(4,2,2*wexp+2)
    sns.tsplot(data=picks, condition=["Exp1","Exp2","Exp3","Exp4"], legend=True, color=['r','g','b','c']);
    #plt.title("100-Step Reward");
    #plt.xlabel("Bandit Timestep");
    #plt.ylabel("# of picks/N"); 
    plt.xlim(0,1500); ax.set_xticks(ax.get_xticks()[::2]); plt.xlim(0,1500);
    plt.ylim(0,1); ax.set_yticks(ax.get_yticks()[::2]); plt.ylim(0,1);             

fig.text(0.5, 0.04, 'Iteration Step', ha='center', va='center')
fig.text(0.06, 0.5, 'Iteration-Normalized Cumulative Reward', ha='center', va='center', rotation='vertical')             
fig.text(0.5, 0.5, 'Iteration-Normalized Number of Calls', ha='center', va='center', rotation='vertical')             

               
plt.show();