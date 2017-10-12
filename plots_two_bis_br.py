# -*- coding: utf-8 -*-
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure();

N = 3#4;
num_of_reps = 3#5;
for wexp in range(3):
    
    exps = ["[20, 10],[20, 35]","[20, 45],[20, 35]","[28, 37],[50, 20]"]#["[10, 30],[20, 50]","[25, 10],[15, 70]","[42, 10],[20, 42]","[10, 10],[60, 20]"];
    data_1 = [];
    data_2 = [];
    r_data = [];
    
    for i in range(1,num_of_reps+1):
        data = pickle.load(open("data/plot_onebis_data/data_"+str(exps[wexp])+"up3_"+str(i)+".pkl", "rb" ))
        #print(str(data[2]))
        data_1.append(np.array(data[0][:-1]));
        #data_1[-1] = data_1[-1][None][None].T
        t = np.array([t for t in range(1,len(data_1[-1])+1)]);
                     
        data_1[-1] = data_1[-1]/t
        data_1[-1] = data_1[-1][None]
    
        tmp = np.concatenate(data[1][:],axis=0)/t[None].T
        data_2.append(tmp[None].T)

    r_data = 0.2;#np.mean(pickle.load(open("data/plot_one_data/experts_100avg_performance_exp"+str(wexp)+".pkl", "rb" )))
    #r_data = np.concatenate([np.array([r_data])]*1500)[None];
    
    #x = np.linspace(1,len(data[0]),num=len(data[0]));
    hund_step_rew = np.concatenate(data_1,axis=0)
    #fixed_cum_rew = np.concatenate(r_data[None],axis=0)
    
    picks = np.concatenate(data_2,axis=2).T

    ax = plt.subplot(3,2,2*wexp+1)
    #ax = plt.subplot(2,4,wexp+1)
    coloritos = ['r','g','b'];
    #sns.tsplot(data=r_data[0], condition=["$\overline{R}_"+str(wexp+1)+"$"], color=coloritos[wexp], linestyle='--');
    sns.tsplot(data=r_data-hund_step_rew, color='k');
    #sns.tsplot(data=fixed_cum_rew, condition=["Fixed Exp. "+str(wexp+1)+" (100-S Avg. Rew)"], legend=True, color='b');       
    plt.xlim(0,1500); ax.set_xticks(ax.get_xticks()[::2]); plt.xlim(0,1500); 
    plt.ylim(0,0.08); ax.set_yticks(ax.get_yticks()[::2]); plt.ylim(0,0.08);
    
    ax = plt.subplot(3,2,2*wexp+2)
    #ax = plt.subplot(2,4,wexp+5)
    sns.tsplot(data=picks, condition=["Exp1","Exp2","Exp3"], legend=True, color=['r','g','b','c']);
    #plt.title("100-Step Reward");
    #plt.xlabel("Bandit Timestep");
    #plt.ylabel("# of picks/N"); 
    plt.xlim(0,1500); ax.set_xticks(ax.get_xticks()[::2]); plt.xlim(0,1500);
    plt.ylim(0,1); ax.set_yticks(ax.get_yticks()[::2]); plt.ylim(0,1);            

fig.text(0.5, 0.04, 'Iteration Step', ha='center', va='center')
fig.text(0.06, 0.5, 'Iteration-Normalized Regret', ha='center', va='center', rotation='vertical')             
fig.text(0.5, 0.5, 'Iteration-Normalized Number of Calls', ha='center', va='center', rotation='vertical')             

plt.show();