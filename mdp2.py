import numpy as np

import matplotlib.pyplot as plt

################################################################################
# Multi-armed bandit strategy class
# N is the number of actions to choose from
# parameters are beta, eta, gamma, as per pg. 160 of Perdition, Buring, Flames
# g is the reward received
# g2 corresponds to g' in the book's notes
################################################################################

class MultiArmedBanditStrategy:
    
    N = None
    beta = None
    eta = None
    gamma = None
    cumulative_reward = None
    
    w = None
    p = None
    
    def __init__(self, N, beta, eta, gamma, cumulative_reward = 0):
        
        self.N = N
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        self.cumulative_reward = 0
        self.ex_cumulative_reward = np.zeros(self.N) #VRR
        self.ex_calls = np.zeros(self.N) + 1e-6
        self.eps = 1.0; #VRR
        
        
        self.w = np.ones(N)
        self.p = self.w / N
        self.bound = np.ones(N)*np.log(2/0.95)
        #self.flag = True;
        
    def calculate_parameters(n, N, delta):
        
        beta = np.sqrt( (1 / (n*N)) * np.log(N/delta) )
        gamma = (4*N*beta) / (3+beta)
        eta = gamma / (2*N)
        
        return beta, eta, gamma
        
    def update(self, action, g, verbose = False):
        
        self.cumulative_reward += g
        
        # Calculate the estimated gains
        g2 = np.ones(self.N) * self.beta / self.p
        
        # Accept if action is a index or if it is a indicator
        if len(action) > 0:
            g2 = g2 + (g * action / self.p)
        else:
            g2[action] = g2[action] + (g / p[action])
        
        # Update the weights
        self.w = self.w * np.exp(self.eta * g2)
        W = np.sum(self.w)
        
        # Calculate the updated probability distribution
        self.p = ((1 - self.gamma) * self.w / W) + (self.gamma / self.N)
        
        if verbose:
            print( 'Cumulative reward: {}'.format(self.cumulative_reward) )
            print( 'New probabilities: {}'.format(self.p) )
            
    def update_1(self, action, g, verbose = False):
        
        self.cumulative_reward += g
        self.ex_cumulative_reward += (g * action);
        self.ex_calls += action;
        # Calculate the estimated gains
        #g2 = np.ones(self.N) * self.beta / self.p
        
        # Accept if action is a index or if it is a indicator
        #if len(action) > 0:
        #    g2 = g2 + (g * action / self.p)
        #else:
        #    g2[action] = g2[action] + (g / p[action])
        
        # Update the weights
        self.w = np.exp(self.eta * self.ex_cumulative_reward)
        W = np.sum(self.w)
        
        # Calculate the updated probability distribution
        self.eps *= 0.998;
        if(np.random.uniform() > self.eps):
            self.p = self.w / W;
        else:
            self.p = np.zeros(self.N);
            self.p[np.random.randint(self.N)] = 1.0;
        
        if verbose:
            print( 'Cumulative reward: {}'.format(self.cumulative_reward) )
            print( 'New probabilities: {}'.format(self.p) )
            
    def update_2(self, action, g, verbose = False): #VRR need to correct algorithm, not 1/n as believed
            
        self.cumulative_reward += g
        self.ex_cumulative_reward += (g * action);
        self.ex_calls += action;
        
        self.w = self.ex_cumulative_reward / self.ex_calls
        self.bound = np.sqrt(np.ones(self.N)*2.0*np.log(2.0/0.05)*(0.1**2) / (self.ex_calls))
        
        # Calculate the updated probability distribution

        self.p = np.zeros(self.N);
        self.p[np.argmax(self.w + self.bound)] = 1.0;
        
        if verbose:
            print( 'Cumulative reward: {}'.format(self.cumulative_reward) )
            print( 'New probabilities: {}'.format(self.p) )  
            
    def update_3(self, action, g, verbose = False): #VRR need to correct algorithm, not 1/n as believed
            
        self.cumulative_reward += g
        self.ex_cumulative_reward += (g * action);
        self.ex_calls += action;
        
        self.w = self.ex_cumulative_reward / self.ex_calls
        
        t = np.sum(self.ex_calls)
        #self.bound = np.sqrt(np.ones(self.N)*8.0*np.log(t+1.0)*0.01 / (self.ex_calls))
        self.bound = np.sqrt(np.ones(self.N)*np.log(t+1.0) / (self.ex_calls))*0.1
        
        # Calculate the updated probability distribution

        self.p = np.zeros(self.N);
        self.p[np.argmax(self.w + self.bound)] = 1.0;
        
        if verbose:
            print( 'Cumulative reward: {}'.format(self.cumulative_reward) )
            print( 'New probabilities: {}'.format(self.p) )             
        
    def draw_action(self):
        return np.random.multinomial(1,self.p)

################################################################################
# Multi-armed bandit test class
# This is a simple class which creates N bandits with Gaussian rewards and
#   equidistant rewards
################################################################################

class MultiArmedBanditTester:
    
    N = None
    
    mu = None
    
    def __init__(self,N):
        self.N = N
        
        mu = np.random.rand(self.N)
        # For ease of use, always make the last action most rewarding
        self.mu = np.sort(mu)
        
    def get_best_action(self):
        return 
        
    # This takes in action as a vector with one 1 entry and rest 0
    def pull_bandit(self,action):
        raw_reward = np.dot(self.mu,action) + (np.random.randn() / N)
        
        return( np.median([0,1,raw_reward]) )

################################################################################
# Main code
################################################################################

#n = 1000
#N = 3
#delta = 0.05
#
#beta, eta, gamma = MultiArmedBanditStrategy.calculate_parameters( n, N, delta )
## print( 'beta = {}, eta = {}, gamma = {}'.format(beta,eta,gamma) )
#
#mabs = MultiArmedBanditStrategy( N, beta, eta, gamma )
#mabt = MultiArmedBanditTester( N )
#
#print( 'Bandits: {}\n'.format(mabt.mu) )
#
#cumulative_rewards = np.zeros(n)
#cumulative_rewards_normalized = np.zeros(n)
#
#for i in range(n):
#    
#    cumulative_rewards[i] = mabs.cumulative_reward
#    if i > 0:
#        cumulative_rewards_normalized[i] = mabs.cumulative_reward / i
#    
#    action = mabs.draw_action()
#    # print( 'Action taken: {}'.format(action) )
#    
#    g = mabt.pull_bandit(action)
#    # print( 'Reward received: {}'.format(g) )
#    
#    mabs.update( action, g )
#    # mabs.update( action, g, verbose = True )
#    # print('')
#    #print(str(mabs.p));
#
#print( 'Final probabilities: {}'.format(mabs.p) )
#print( 'Cumulative rewards: {}'.format(mabs.cumulative_reward) )
#
## # Plot cumulative rewards
## # Calculate approximate best action
## x = np.linspace(0,n)
## y = np.max(mabt.mu) * x
##
## plt.plot(x,y)
## plt.plot(cumulative_rewards)
##
## plt.legend( ['best action (estimated)', 'taken action'] )
## plt.title('cumulative rewards')
#
## Plot normalized cumulative rewards
## Calculate approximate best action
#x = np.linspace(0,n)
#y = np.max(mabt.mu) * np.ones(len(x))
#
#plt.plot(x,y)
#plt.plot(cumulative_rewards_normalized)
#
#plt.legend( ['best action (estimated)', 'taken action'] )
#plt.title('cumulative rewards (normalized)')
#
#plt.show()

