import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time

import dqn
from dqn_utils import *
from atari_wrappers import *

import pickle
from mdp import MultiArmedBanditStrategy
import matplotlib.pyplot as plt

def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def atari_learn(env,
                session,
                num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        q_func=atari_model, #just the neural network as defined above
        optimizer_spec=optimizer, #just a named tuple containing the grad alg, lr_shedule etc..
        session=session,  #The tf session 
        exploration=exploration_schedule, #epsilon greedy schedule
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000, #Replay buffer size... size of what?
        batch_size=32, #Gradient ascent batch size I guess..
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/Users/cusgadmin/Documents/Berkeley/Research/Tomlin/RL/homework/hw3/Videos'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env     
    
def main():
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    # VRR: We can use this to try multiple games. (0) Flat 3d space invaders (1) Breakout  (2) Racing Car (3) Pong (4) Weird pyramid (5) Seaquest (6) Old looking space invaders
    task = benchmark.tasks[5]

    # Run training
    seed = round(time.time())# Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)
    
#    #["[10, 30]_[20, 50]","[25, 10]_[15, 70]","[42, 10]_[20, 42]","[10, 10]_[60, 20]"]
#    #origin = [[10,30],[25,10],[42,10],[10,10]]
#    #h_and_w = [[20,50],[15,70],[20,42],[60,20]]
#    origin = [[50,10],[10,50],[40,20],[10,15]]
#    h_and_w = [[15,70],[60,20],[20,50],[20,42]]
#    
#    obs = sensor_noise(env.reset(),origin[0][0],origin[0][1],h_and_w[0][0],h_and_w[0][1])
#    obs = sensor_noise(obs,origin[1][0],origin[1][1],h_and_w[1][0],h_and_w[1][1])
#    obs = sensor_noise(obs,origin[2][0],origin[2][1],h_and_w[2][0],h_and_w[2][1])
#    obs = sensor_noise(obs,origin[3][0],origin[3][1],h_and_w[3][0],h_and_w[3][1])
#    plt.imshow(obs[:,:,0])
#    plt.pause(10.0)
#    env.reset();        

    session = get_session()
    #atari_learn(env, session, num_timesteps=task.max_timesteps/2)
    robust_rl(env, session);
    #robust_rl3(env, session);
    #avore = expert_eval(env,session);
    #print(str(avore));

if __name__ == "__main__":
    main()
