import tensorflow as tf
import numpy as np
import gym
import os
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

env = gym.make("MsPacman-v0")

# Preprocessing for images of the game(observation - images of screen')
mspacman_color = np.array([210,164,74]).mean()

def preprocess_observation(obs):
    ''' Actual preprocessing '''
    # Cutting an image(making int smaller)
    img = obs[1:176:2,::2]
    # Greyscaling it
    img = img.mean(axis=2)
    # Contrast
    img[img==mspacman_color] = 0
    # Normalizing between -1 and 1
    img = (img - 128)/128 - 1
    # Returning new image
    return img.reshape(88,80,1)

# We are going to create 2 DQN-networks(DeepMind analogy)
# That's why we need a function for this:
input_height = 88
input_width = 80
input_channels = 1
conv_n_maps = [32,64,64]
conv_kernel_sizes = [(8,8),(4,4),(3,3)]
conv_strides = [4,2,1]
conv_paddings = ["SAME"] * 3
conv_activation = [tf.nn.relu] * 3
# Conv3 has 64 feature maps of size 11x10
n_hidden_in = 64*11*10
n_hidden = 512
hidden_activation = tf.nn.relu
# 9 discrete actions are possible
n_outputs = env.action_space.n
initializer = tf.contrib.layers.variance_scaling_initializer()

def q_network(X_state,name):
    ''' Function to build a particular DQN architecture '''
    prev_layer = X_state
    with tf.variable_scope(name) as scope:
        for n_maps,kernel_size,strides,padding,activation in zip(
                conv_n_maps,conv_kernel_sizes,conv_strides,
                conv_paddings,conv_activation):
            prev_layer = tf.layers.conv2d(
                    prev_layer,filters=n_maps,kernel_size=kernel_size,
                    strides=strides,padding=padding,activation=activation,
                    kernel_initializer=initializer)
        last_conv_layer_flat = tf.reshape(prev_layer,
                                        shape=[-1,n_hidden_in])
        hidden = tf.layers.dense(last_conv_layer_flat,n_hidden,
                    activation=hidden_activation,
                    kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden,n_outputs,
                    kernel_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                            scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]:var for var in trainable_vars}
    return outputs,trainable_vars_by_name
        
X_state = tf.placeholder(tf.float32,shape=[None,input_height,
                                            input_width,input_channels])
online_q_values,online_vars = q_network(X_state,name="q_networks/online")
target_q_values,target_vars = q_network(X_state,name="q_networks/target")
copy_ops = [target_var.assign(online_vars[var_name])
        for var_name,target_var in target_vars.items()]
copy_online_to_target = tf.group(copy_ops)

# Training 
X_action = tf.placeholder(tf.int32,shape=[None])
q_value = tf.reduce_sum(target_q_values * tf.one_hot(X_action,
                n_outputs),axis=1,keep_dims=True)
y = tf.placeholder(tf.float32,shape=[None,1])
error = tf.abs(y - q_value)
clipped_error = tf.clip_by_value(error,0.0,1.0)
linear_error = 2 * (error - clipped_error)
loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

learning_rate = 0.001
momentum = 0.95

global_step = tf.Variable(0,trainable=False,name='global-step')
optimizer = tf.train.MomentumOptimizer(learning_rate,momentum,
                                            use_nesterov=True)
training_op = optimizer.minimize(loss,global_step=global_step)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

from collections import deque

replay_memory_size = 500000
replay_memory = deque([],maxlen=replay_memory_size)

def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    # State,action,reward,next state, etc.
    cols = [[],[],[],[],[]]
    for idx in indices:
        memory = replay_memory[idx]
        for col,value in zip(cols,memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0],cols[1],cols[2].reshape(-1,1),cols[3],
            cols[4].reshape(-1,1))

eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 2000000

def epsilon_greedy(q_values,step):
    epsilon = max(eps_min,eps_max - (eps_max-eps_min)
            * step/eps_decay_steps)
    # Random action
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    # Optimal action
    else:
        return np.argmax(q_values)

# Stage of implementation
# Number of steps
n_steps = 4000000
# Starting learning after 10000 games
training_start = 10000
# Make step of learning every 4 iteration 
training_interval = 4
# Save every 1000
save_steps = 1000
# Copy dynamic DQN in target DQN every 10000 steps of learning
copy_steps = 10000
discount_rate = 0.99
skip_start = 90
batch_size = 50
iteration = 0
checkpoint_path = "./my_dqn.ckpt"
# Just to drop the game
done = True

'''
# Main part
with tf.Session() as sess:
    if os.path.isfile(checkpoint_path + ".index"):
        saver.restore(sess,checkpoint_path)
    else:
        init.run()
        copy_online_to_target.run()
    while True:
        step = global_step.eval()
        if step >= n_steps:
            break
        iteration += 1
        # Repeating the game
        if done:
            obs = env.reset()
            for skip in range(skip_start):
                obs,reward,done,info = env.step(0)
            state = preprocess_observation(obs)
        # Dynamic DQN estimate what to do
        q_values = online_q_values.eval(feed_dict={X_state:[state]})
        action = epsilon_greedy(q_values,step)
        # Dynamic DQN plays
        obs,reward,done,info = env.step(action)
        next_state = preprocess_observation(obs)
        # Remember what just happened
        replay_memory.append((state,action,reward,next_state,1.0 - done))
        state = next_state
        if iteration < training_start or iteration%training_interval != 0:
            # train only after a "warm-up"
            continue
        # Choose remembered data and use target DQN
        # for creating target Q-value
        X_state_val,X_action_val,rewards,X_next_state_val,continues = (sample_memories(batch_size))
        next_q_values = target_q_values.eval(
                feed_dict={X_state:X_next_state_val})
        max_next_q_values = np.max(next_q_values,axis=1,keepdims=True)
        y_val = rewards + continues + discount_rate * max_next_q_values
        # Train dynamic DQN
        training_op.run(feed_dict={X_state:X_state_val,
                                    X_action:X_action_val,y:y_val})
        # Regularly copy dynamic DQN in target DQN 
        if step%copy_steps == 0:
            copy_online_to_target.run()
        # And regularly save
        if step%save_steps == 0:
            saver.save(sess,checkpoint_path)
            print('Saving a model on a current step:{}'.format(step))
'''
# Playing stage
frames = []
n_max_steps = 10000

with tf.Session() as sess:
    saver.restore(sess,checkpoint_path)
    obs = env.reset()
    for step in range(n_max_steps):
        state = preprocess_observation(obs)
        # what to do
        q_values = online_q_values.eval(feed_dict={X_state:[state]})
        action = np.argmax(q_values)
        obs,reward,done,info = env.step(action)
        img = env.render(mode="rgb_array")
        frames.append(img)
        
        if done:
            break

def update_scene(num,frames,patch):
    patch.set_data(frames[num])
    return patch

def plot_animation(frames,repeat=False,interval=40):
    ''' plotting game process '''
    plt.close()
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(fig,update_scene,fargs=(frames,patch),
                            frames=len(frames),repeat=repeat,interval=interval)

# plotting pac-man
video = plot_animation(frames)
plt.show()
       






