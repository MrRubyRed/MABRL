"""This file includes a collection of utility functions that are useful for
implementing DQN."""
import gym
import tensorflow as tf
import numpy as np
import random
from run_dqn_atari import atari_model, get_session
import pickle
from mdp import MultiArmedBanditStrategy
import time

def sensor_noise(obs,pos_i,pos_j,l,w):
    obs[pos_i:pos_i+l,pos_j:pos_j+w,0] = 0;
    return obs;

def huber_loss(x, delta=1.0):
    # https://en.wikipedia.org/wiki/Huber_loss
    return tf.select(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )

def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()

class ConstantSchedule(object):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v

def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)

class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

def compute_exponential_averages(variables, decay):
    """Given a list of tensorflow scalar variables
    create ops corresponding to their exponential
    averages
    Parameters
    ----------
    variables: [tf.Tensor]
        List of scalar tensors.
    Returns
    -------
    averages: [tf.Tensor]
        List of scalar tensors corresponding to averages
        of al the `variables` (in order)
    apply_op: tf.runnable
        Op to be run to update the averages with current value
        of variables.
    """
    averager = tf.train.ExponentialMovingAverage(decay=decay)
    apply_op = averager.apply(variables)
    return [averager.average(v) for v in variables], apply_op

def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients)

def initialize_interdependent_variables(session, vars_list, feed_dict):
    """Initialize a list of variables one at a time, which is useful if
    initialization of some variables depends on initialization of the others.
    """
    vars_left = vars_list
    while len(vars_left) > 0:
        new_vars_left = []
        for v in vars_left:
            try:
                # If using an older version of TensorFlow, uncomment the line
                # below and comment out the line after it.
                #session.run(tf.initialize_variables([v]), feed_dict)
                session.run(tf.variables_initializer([v]), feed_dict)
            except tf.errors.FailedPreconditionError:
                new_vars_left.append(v)
        if len(new_vars_left) >= len(vars_left):
            # This can happend if the variables all depend on each other, or more likely if there's
            # another variable outside of the list, that still needs to be initialized. This could be
            # detected here, but life's finite.
            raise Exception("Cycle in variable dependencies, or external precondition unsatisfied.")
        else:
            vars_left = new_vars_left

def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)

class ReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs      = None
        self.action   = None
        self.reward   = None
        self.done     = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 2)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done

#VRR: Added functions
def expert_eval(env, session):

    N = 7;
 
    frame_history_len = 4;
    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c) #the input (i.e. the observation) is a small sequence of images 
    num_actions = env.action_space.n
    
    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph              = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph            = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph          = tf.placeholder(tf.float32, [None])    
    
    # casting to float on GPU ensures lower data transfer times.
    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0
    


    #Load each neural network expert and create place holders (TODO)
    Q = [];
    opt_a = [];
    q_func_vars = [];
    model = [];
    occl_list = ["[10, 30]_[20, 50]","[25, 10]_[15, 70]","[42, 10]_[20, 42]","[10, 10]_[60, 20]","[0, 0]_[0, 0]","[-2, -2]_[-2, -2]","[-3, -3]_[-3, -3]"];#["[10, 30]_[20, 50]","[25, 10]_[15, 70]","[42, 10]_[20, 42]"];
    if len(occl_list) != N: 
        print('Error, different number of occlusions and experts.')
        return -1;
    for i in range(N):
        Q.append(atari_model(obs_t_float, num_actions, scope="q_func"+str(i), reuse=False)) 
        opt_a.append(tf.argmax(Q[-1],1));
        q_func_vars.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func'+str(i)))
        model.append(pickle.load(open("experts/expert_np_"+occl_list[i]+".pkl", "rb")))
   
    #initialize_interdependent_variables(session, tf.global_variables(), {obs_t_ph: obs_t_batch,obs_tp1_ph: obs_tp1_batch,});
    #sess = tf.Session();
    init = tf.initialize_all_variables();
    session.run(init);


    for i in range(N):        
        update_target_fn = []
        for var, var_target in zip(q_func_vars[i],model[i]):
            update_target_fn.append(var.assign(var_target))
        update_target_fn = tf.group(*update_target_fn)
        session.run(update_target_fn);



    
    #Create replay-buffer to use the functions within.
    replay_buffer = ReplayBuffer(1000000, 4);



    ### NOISE PARAMS ###
    origin = [20,10]
    h_and_w = [20,35]
    #O1:[50,10],[15,70] ~ O2:[10,50],[60,20] ~ O3:[40,20],[20,50] ~ O4:[10,15],[20,42]
    ####################
    run = 1;



    last_obs = sensor_noise(env.reset(),origin[0],origin[1],h_and_w[0],h_and_w[1]);
                            
    import matplotlib.pyplot as plt
    #plt.imshow(last_obs[:,:,0]);
    #plt.pause(0.2);
                            
    rew = [0]*N; picks = [0+1e-6]*N; crew = 0;
    runs = 50*100;#1500*100;
    it = []; rec1 = []; rec2 = []; rec3 = []; rec4 = []; rec5 = [];
    prob_hist = np.zeros((N,runs));                       
    for i in range(runs):
        
        indx = replay_buffer.store_frame(last_obs);
        p_obs = replay_buffer.encode_recent_observation();
        
        len_ep = 100.0;
        if i % len_ep == 0:
            expert_ = 2;
#            if(i >= 0 and i < runs/2):
#                expert_ = 0;
#            elif(i >= runs/2 and i < runs):
#                expert_ = 5;            
#            expert_ = np.random.randint(N);
#            if(i >= 0 and i < runs/N):
#                expert_ = 0;
#            elif(i >= runs/N and i < 2*runs/N):
#                expert_ = 1;
#            elif(i >= 2*runs/N and i < 3*runs/N):
#                expert_ = 2; 
#            elif(i >= 3*runs/N and i < 4*runs/N):
#                expert_ = 3;
#            expert_ = 3;
            print(str(i) + " || " + str(np.array([rew[0]/picks[0],rew[1]/picks[1],rew[2]/picks[2],rew[3]/picks[3],rew[4]/picks[4],rew[5]/picks[5],rew[6]/picks[6]])) + " || " + str(round(picks[0]))+","+ str(round(picks[1]))+","+ str(round(picks[2]))+","+ str(round(picks[3])));
            it.append(i);
            #rec1.append(rew[0]/picks[0]);
            #rec2.append(rew[1]/picks[1]);
            #rec3.append(rew[2]/picks[2]);
            #rec4.append(rew[3]/picks[3]);
            rec5.append(rew[expert_]/picks[expert_]);
            picks[expert_] += 1.0;
        
        action = session.run(opt_a[expert_],{obs_t_ph:p_obs[None],});
        obs, reward, done, info = env.step(action);
        replay_buffer.store_effect(indx,action,reward,done);
        reward = reward/len_ep;
        rew[expert_] += reward;
        crew += reward;

        if(done == False):
            #last_obs = obs;#sensor_noise(obs,0,0,img_h,42);
            last_obs = sensor_noise(obs,origin[0],origin[1],h_and_w[0],h_and_w[1]);
        else:
            last_obs = sensor_noise(env.reset(),origin[0],origin[1],h_and_w[0],h_and_w[1]);
                                    
    #plt.plot(it,rec1);
    #plt.plot(it,rec2);
    #plt.plot(it,rec3);
    #plt.plot(it,rec4);
    #plt.plot(it,rec5);    

    for _ in range(1500-50): rec5.append(rec5[-1]);
    #pickle.dump([rec5], open("data/plot_two_data/experts_100avg_performance_exp_"+str(expert_)+"_"+str(run)+"_"+".pkl", "wb" ) )
    
    return np.array([rew[0],rew[1]]);

def robust_rl(env, session):

    time1 = time.time();
    # Create MultiArmedBanditStrategy
    n = 1500;
    N = 3#4;
    delta = 0.01;
    beta, eta, gamma = MultiArmedBanditStrategy.calculate_parameters( n, N, delta );
    print( 'beta = {}, eta = {}, gamma = {}'.format(beta,eta,gamma) );
    
    mabs = MultiArmedBanditStrategy( N, 0.0, eta*10.0, gamma );  
    #mabs = MultiArmedBanditStrategy( N, 0.0, 0.6, 0.5 ); 
 
    frame_history_len = 4;
    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c) #the input (i.e. the observation) is a small sequence of images 
    num_actions = env.action_space.n
    
    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph              = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph            = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph          = tf.placeholder(tf.float32, [None])    
    
    # casting to float on GPU ensures lower data transfer times.
    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0
    


    #Load each neural network expert and create place holders (TODO)
    Q = [];
    opt_a = [];
    q_func_vars = [];
    model = [];
    occl_list = ["[20, 10]_[20, 35]","[20, 45]_[20, 35]","[28, 37]_[50, 20]"]#["[10, 30]_[20, 50]","[25, 10]_[15, 70]","[42, 10]_[20, 42]","[10, 10]_[60, 20]"];#["[10, 10]_[60, 20]","[10, 50]_[60, 20]"];
    if len(occl_list) != N: 
        print('Error, different number of occlusions and experts.')
        return -1;
    for i in range(N):
        Q.append(atari_model(obs_t_float, num_actions, scope="q_func"+str(i), reuse=False)) 
        opt_a.append(tf.argmax(Q[-1],1));
        q_func_vars.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func'+str(i)))
        #model.append(pickle.load(open("experts/expert_np_"+occl_list[i]+".pkl", "rb")))
        model.append(pickle.load(open("experts/b_expert_np_"+occl_list[i]+".pkl", "rb")))
        
    #initialize_interdependent_variables(session, tf.global_variables(), {obs_t_ph: obs_t_batch,obs_tp1_ph: obs_tp1_batch,});
    #sess = tf.Session();
    init = tf.initialize_all_variables();
    session.run(init);

    for i in range(N):        
        update_target_fn = []
        for var, var_target in zip(q_func_vars[i],model[i]):
            update_target_fn.append(var.assign(var_target))
        update_target_fn = tf.group(*update_target_fn)
        session.run(update_target_fn);

   
    #Create replay-buffer to use the functions within.
    replay_buffer = ReplayBuffer(1000000, 4);


    ### OCCLUSION PARAMS ###
    origin = [10,54]#[28,27]#[10,10],[10,54],
    h_and_w = [45,20]#[40,30]#[45,20],[45,20],    
    #origin = [40,20]#[50,10]#[10,50]#[40,20]#[10,15]
    #h_and_w = [20,50]#[15,70]#[60,20]#[20,50]#[20,42]
    ####################
    etype = 3;
    rep = 3;
    
    stwu = env.reset();
    last_obs = sensor_noise(stwu,origin[0],origin[1],h_and_w[0],h_and_w[1]);
    #others = sensor_noise(last_obs,10,30,20,50);
    #others = sensor_noise(others,42,10,20,42);                       
    import matplotlib.pyplot as plt
    plt.imshow(last_obs[:,:,0]);
    plt.pause(0.2);
                            
    rew = [0]*N; picks = [0+1e-6]*N; crew = 0;
    runs = n*100;
    
    regr = [0.0];
    calls = [];    
                       
    for i in range(runs):
        
        indx = replay_buffer.store_frame(last_obs);
        p_obs = replay_buffer.encode_recent_observation();
        
        len_ep = 100.0;
        if i % len_ep == 0.0:
            
            regr.append(crew + regr[-1]);
            calls.append(np.array(picks)[None]);

            if(i is not 0 and etype == 1): mabs.update(expert,crew);
            if(i is not 0 and etype == 2): mabs.update_1(expert,crew);
            if(i is not 0 and etype == 3): mabs.update_2(expert,crew);
            if(i is not 0 and etype == 4): mabs.update_3(expert,crew);
            crew = 0.0;
            #mabs.gamma = mabs.gamma*0.99;    
            #print(str(mabs.p) + " | actions: " + str(action1) +","+ str(action2) +","+ str(action3) + "| w = " + str(mabs.w) + " | REWS = " + str(round(rew1,2))+","+str(round(rew2,2))+","+str(round(rew3,2)));
            print(str(mabs.p) + "| w,b = " + str(mabs.w)+str(mabs.bound) + " | REWS = " + 
                  str(i)#+str(round(rew[3]/picks[3],3))+" | PICKS = "
                  #str(round(picks[0],2))+","+str(round(picks[1],2))+","+str(round(picks[2],2))+","+str(round(picks[3],3))
                  );            
            expert = mabs.draw_action();
            expert_ = np.where(expert)[0][0]
            picks[expert_] += 1.0;
        
        action = session.run(opt_a[expert_],{obs_t_ph:p_obs[None],});
        obs, reward, done, info = env.step(action);
        replay_buffer.store_effect(indx,action,reward,done);
        reward = reward/len_ep;
        rew[expert_] += reward;
        crew += reward;

        if(done == False):
            #last_obs = obs;#sensor_noise(obs,0,0,img_h,42);
            last_obs = sensor_noise(obs,origin[0],origin[1],h_and_w[0],h_and_w[1]);
        else:
            #last_obs = env.reset();#sensor_noise(env.reset(),0,0,img_h,42);
            last_obs = sensor_noise(env.reset(),origin[0],origin[1],h_and_w[0],h_and_w[1]);                            

    time2 = time.time();                                
                                    
    #if(etype == 1): pickle.dump([regr,calls], open("data/data_"+str(origin)+","+str(h_and_w)+"up1"+".pkl", "wb" ) )
    #if(etype == 2): pickle.dump([regr,calls], open("data/data_"+str(origin)+","+str(h_and_w)+"up2"+".pkl", "wb" ) )
    #if(etype == 3): pickle.dump([regr,calls,time2-time1], open("data/plot_twobis_data/data_"+str(origin)+","+str(h_and_w)+"up3_"+str(rep)+".pkl", "wb" ) )
    #if(etype == 4): pickle.dump([regr,calls,time2-time1], open("data/plot_onebis_data/data_"+str(origin)+","+str(h_and_w)+"up4_"+str(rep)+".pkl", "wb" ) )                    
                    
def robust_rl2(env, session):

    N = 4;
    occls = [[[10,30],[20,50]],[[25,10],[15,70]],[[42,10],[20,42]],[[10,10],[60,20]]];
    for k in range(N):
        _ = env.reset();
        for z in range(5):
            time1 = time.time();
            # Create MultiArmedBanditStrategy
            n = 1500;
            delta = 0.01;
            beta, eta, gamma = MultiArmedBanditStrategy.calculate_parameters( n, N, delta );
            print( 'beta = {}, eta = {}, gamma = {}'.format(beta,eta,gamma) );
            
            mabs = MultiArmedBanditStrategy( N, 0.0, eta*10.0, gamma );  
            #mabs = MultiArmedBanditStrategy( N, 0.0, 0.6, 0.5 ); 
         
            frame_history_len = 4;
            if len(env.observation_space.shape) == 1:
                # This means we are running on low-dimensional observations (e.g. RAM)
                input_shape = env.observation_space.shape
            else:
                img_h, img_w, img_c = env.observation_space.shape
                input_shape = (img_h, img_w, frame_history_len * img_c) #the input (i.e. the observation) is a small sequence of images 
            num_actions = env.action_space.n
            
            # set up placeholders
            # placeholder for current observation (or state)
            obs_t_ph              = tf.placeholder(tf.uint8, [None] + list(input_shape))
            # placeholder for current action
            act_t_ph              = tf.placeholder(tf.int32,   [None])
            # placeholder for current reward
            rew_t_ph              = tf.placeholder(tf.float32, [None])
            # placeholder for next observation (or state)
            obs_tp1_ph            = tf.placeholder(tf.uint8, [None] + list(input_shape))
            # placeholder for end of episode mask
            # this value is 1 if the next state corresponds to the end of an episode,
            # in which case there is no Q-value at the next state; at the end of an
            # episode, only the current state reward contributes to the target, not the
            # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
            done_mask_ph          = tf.placeholder(tf.float32, [None])    
            
            # casting to float on GPU ensures lower data transfer times.
            obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
            obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0
            
        
        
            #Load each neural network expert and create place holders (TODO)
            Q = [];
            opt_a = [];
            q_func_vars = [];
            model = [];
            occl_list = ["[10, 30]_[20, 50]","[25, 10]_[15, 70]","[42, 10]_[20, 42]","[10, 10]_[60, 20]"];#["[10, 10]_[60, 20]","[10, 50]_[60, 20]"];
            if len(occl_list) != N: 
                print('Error, different number of occlusions and experts.')
                return -1;
            for i in range(N):
                Q.append(atari_model(obs_t_float, num_actions, scope="q_func"+str(i), reuse=False)) 
                opt_a.append(tf.argmax(Q[-1],1));
                q_func_vars.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func'+str(i)))
                model.append(pickle.load(open("experts/expert_np_"+occl_list[i]+".pkl", "rb")))
           
            #initialize_interdependent_variables(session, tf.global_variables(), {obs_t_ph: obs_t_batch,obs_tp1_ph: obs_tp1_batch,});
            #sess = tf.Session();
            init = tf.initialize_all_variables();
            session.run(init);
        
            for i in range(N):        
                update_target_fn = []
                for var, var_target in zip(q_func_vars[i],model[i]):
                    update_target_fn.append(var.assign(var_target))
                update_target_fn = tf.group(*update_target_fn)
                session.run(update_target_fn);
        
           
            #Create replay-buffer to use the functions within.
            replay_buffer = ReplayBuffer(1000000, 4);
        
        
            ### OCCLUSION PARAMS ###
            origin = occls[k][0]#[50,10]#[10,50]#[40,20]#[10,15]
            h_and_w = occls[k][1]#[15,70]#[60,20]#[20,50]#[20,42]
            ####################
            etype = 4;
            rep = z;
            
            stwu = env.reset();
            last_obs = sensor_noise(stwu,origin[0],origin[1],h_and_w[0],h_and_w[1]);
            #others = sensor_noise(last_obs,10,30,20,50);
            #others = sensor_noise(others,42,10,20,42);                       
            import matplotlib.pyplot as plt
            plt.imshow(last_obs[:,:,0]);
            plt.pause(0.2);
                                    
            rew = [0]*N; picks = [0+1e-6]*N; crew = 0;
            runs = n*100;
            
            regr = [0.0];
            calls = [];    
                               
            for i in range(runs):
                
                indx = replay_buffer.store_frame(last_obs);
                p_obs = replay_buffer.encode_recent_observation();
                
                len_ep = 100.0;
                if i % len_ep == 0.0:
                    
                    regr.append(crew + regr[-1]);
                    calls.append(np.array(picks)[None]);
        
                    if(i is not 0 and etype == 1): mabs.update(expert,crew);
                    if(i is not 0 and etype == 2): mabs.update_1(expert,crew);
                    if(i is not 0 and etype == 3): mabs.update_2(expert,crew);
                    if(i is not 0 and etype == 4): mabs.update_3(expert,crew);
                    crew = 0.0;
                    #mabs.gamma = mabs.gamma*0.99;    
                    #print(str(mabs.p) + " | actions: " + str(action1) +","+ str(action2) +","+ str(action3) + "| w = " + str(mabs.w) + " | REWS = " + str(round(rew1,2))+","+str(round(rew2,2))+","+str(round(rew3,2)));
                    print(str(mabs.p) + "| w,b = " + str(mabs.w)+str(mabs.bound) + " | REWS = " + 
                          str(round(rew[0]/picks[0],3))+","+str(round(rew[1]/picks[1],3))+","+str(round(rew[2]/picks[2],3)) +","+str(round(rew[3]/picks[3],3))+" | PICKS = "
                          #str(round(picks[0],2))+","+str(round(picks[1],2))+","+str(round(picks[2],2))+","+str(round(picks[3],3))
                          );            
                    expert = mabs.draw_action();
                    expert_ = np.where(expert)[0][0]
                    picks[expert_] += 1.0;
                
                action = session.run(opt_a[expert_],{obs_t_ph:p_obs[None],});
                obs, reward, done, info = env.step(action);
                replay_buffer.store_effect(indx,action,reward,done);
                reward = reward/len_ep;
                rew[expert_] += reward;
                crew += reward;
        
                if(done == False):
                    #last_obs = obs;#sensor_noise(obs,0,0,img_h,42);
                    last_obs = sensor_noise(obs,origin[0],origin[1],h_and_w[0],h_and_w[1]);
                else:
                    #last_obs = env.reset();#sensor_noise(env.reset(),0,0,img_h,42);
                    last_obs = sensor_noise(env.reset(),origin[0],origin[1],h_and_w[0],h_and_w[1]);                            
        
            time2 = time.time();                                
                                            
            #if(etype == 1): pickle.dump([regr,calls], open("data/data_"+str(origin)+","+str(h_and_w)+"up1"+".pkl", "wb" ) )
            #if(etype == 2): pickle.dump([regr,calls], open("data/data_"+str(origin)+","+str(h_and_w)+"up2"+".pkl", "wb" ) )
            #if(etype == 3): pickle.dump([regr,calls,time2-time1], open("data/plot_two_data/data_"+str(origin)+","+str(h_and_w)+"up3_"+str(rep)+".pkl", "wb" ) )
            #if(etype == 4): pickle.dump([regr,calls,time2-time1], open("data/plot_onebis_data/data_"+str(origin)+","+str(h_and_w)+"up4_"+str(rep)+".pkl", "wb" ) )
        
        session.close();
        session = get_session()
        
def robust_rl3(env, session): # MAIN FUNCTION, USED FOR PAPER

    N = 4; #number of experts

    time1 = time.time();
    # Create MultiArmedBanditStrategy
    n = 1500; #number of pulls
    delta = 0.01;
    beta, eta, gamma = MultiArmedBanditStrategy.calculate_parameters( n, N, delta );
    print( 'beta = {}, eta = {}, gamma = {}'.format(beta,eta,gamma) );
    
    mabs = MultiArmedBanditStrategy( N, 0.0, eta*10.0, gamma );  
    #mabs = MultiArmedBanditStrategy( N, 0.0, 0.6, 0.5 ); 
 
    frame_history_len = 4;
    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c) #the input (i.e. the observation) is a small sequence of images 
    num_actions = env.action_space.n
    
    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph              = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph            = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph          = tf.placeholder(tf.float32, [None])    
    
    # casting to float on GPU ensures lower data transfer times.
    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0
    


    #Load each neural network expert and create place holders (TODO)
    Q = [];
    opt_a = [];
    q_func_vars = [];
    model = [];
    occl_list = ["[10, 30]_[20, 50]","[25, 10]_[15, 70]","[42, 10]_[20, 42]","[10, 10]_[60, 20]"];#["[10, 10]_[60, 20]","[10, 50]_[60, 20]"];
    if len(occl_list) != N: 
        print('Error, different number of occlusions and experts.')
        return -1;
    for i in range(N):
        Q.append(atari_model(obs_t_float, num_actions, scope="q_func"+str(i), reuse=False)) 
        opt_a.append(tf.argmax(Q[-1],1));
        q_func_vars.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func'+str(i)))
        model.append(pickle.load(open("experts/expert_np_"+occl_list[i]+".pkl", "rb")))
   
    #initialize_interdependent_variables(session, tf.global_variables(), {obs_t_ph: obs_t_batch,obs_tp1_ph: obs_tp1_batch,});
    #sess = tf.Session();
    init = tf.initialize_all_variables();
    session.run(init);

    for i in range(N):        
        update_target_fn = []
        for var, var_target in zip(q_func_vars[i],model[i]):
            update_target_fn.append(var.assign(var_target))
        update_target_fn = tf.group(*update_target_fn)
        session.run(update_target_fn);

   
    #Create replay-buffer to use the functions within.
    replay_buffer = ReplayBuffer(1000000, 4);


    ### OCCLUSION PARAMS ###
    origin = [42,10]#[10,50]#[40,20]#[10,15]
    h_and_w = [20,42]#[60,20]#[20,50]#[20,42]
    ####################
    etype = 4; #what type of update to use from mdp.py file
    
    stwu = env.reset();
    last_obs = sensor_noise(stwu,origin[0],origin[1],h_and_w[0],h_and_w[1]);
    #others = sensor_noise(last_obs,10,30,20,50);
    #others = sensor_noise(others,42,10,20,42);                       
    import matplotlib.pyplot as plt
    plt.imshow(last_obs[:,:,0]);
    plt.pause(0.2);
                            
    rew = [0]*N; picks = [0+1e-6]*N; crew = 0;
    runs = n*100; #number of mdp steps to run (Actual atari steps)
    
    regr = [0.0];
    calls = [];    
                       
    for i in range(runs):
        
        indx = replay_buffer.store_frame(last_obs);
        p_obs = replay_buffer.encode_recent_observation();
        
        len_ep = 200.0;
        if i % len_ep == 0.0:
            
            regr.append(crew + regr[-1]);
            calls.append(np.array(picks)[None]);

            if(i is not 0 and etype == 1): mabs.update(expert,crew);
            if(i is not 0 and etype == 2): mabs.update_1(expert,crew);
            if(i is not 0 and etype == 3): mabs.update_2(expert,crew);
            if(i is not 0 and etype == 4): mabs.update_3(expert,crew);
            crew = 0.0;
            #mabs.gamma = mabs.gamma*0.99;    
            #print(str(mabs.p) + " | actions: " + str(action1) +","+ str(action2) +","+ str(action3) + "| w = " + str(mabs.w) + " | REWS = " + str(round(rew1,2))+","+str(round(rew2,2))+","+str(round(rew3,2)));
            print(str(mabs.p) + "| w,b = " + str(mabs.w)+str(mabs.bound) + " | PICKS = " + 
                  str(mabs.ex_calls/sum(mabs.ex_calls))+" | "+ str(i)
                  #str(round(picks[0],2))+","+str(round(picks[1],2))+","+str(round(picks[2],2))+","+str(round(picks[3],3))
                  );            
            expert = mabs.draw_action(); #expert to be selected
            expert_ = np.where(expert)[0][0] #expert vector to expert's index
            picks[expert_] += 1.0; #update number of calls to that expert
        
        action = session.run(opt_a[expert_],{obs_t_ph:p_obs[None],});
        obs, reward, done, info = env.step(action);
        replay_buffer.store_effect(indx,action,reward,done);
        reward = reward/len_ep;
        rew[expert_] += reward;
        crew += reward;

        if(done == False): #reset environment if we get to a terminal state
            #last_obs = obs;#sensor_noise(obs,0,0,img_h,42);
            last_obs = sensor_noise(obs,origin[0],origin[1],h_and_w[0],h_and_w[1]);
        else:
            #last_obs = env.reset();#sensor_noise(env.reset(),0,0,img_h,42);
            last_obs = sensor_noise(env.reset(),origin[0],origin[1],h_and_w[0],h_and_w[1]);                            

    time2 = time.time();                                
                                    
    #if(etype == 1): pickle.dump([regr,calls], open("data/data_"+str(origin)+","+str(h_and_w)+"up1"+".pkl", "wb" ) )
    #if(etype == 2): pickle.dump([regr,calls], open("data/data_"+str(origin)+","+str(h_and_w)+"up2"+".pkl", "wb" ) )
    #if(etype == 3): pickle.dump([regr,calls,time2-time1], open("data/plot_two_data/data_"+str(origin)+","+str(h_and_w)+"up3_"+str(rep)+".pkl", "wb" ) )
    #if(etype == 4): pickle.dump([regr,calls,time2-time1], open("data/plot_onebis_data/data_"+str(origin)+","+str(h_and_w)+"up4_"+str(rep)+".pkl", "wb" ) )

    session.close();
    session = get_session()        