import gym
import tensorflow as tf
import numpy as np

# Creating our first cart-pole environment!
env = gym.make("CartPole-v0")
'''
BASIC POLICY ! ! !
# Making an observation
obs = env.reset()
def basic_policy(obs):
    angle = obs[2]
    return0 if angle < 0 else 1

totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(1000):
        action = basic_policy(obs)
        obs,reward,done,info = env.step(action)
        episode_rewards += reward
        env.render()
        if done:
            break
    totals.append(episode_rewards)
print(totals)
print(max(totals))
'''

# 4 inputs(4 dimensions in obs vector).Output returns probability of going left
# While probability of going right is 1 - p
n_inputs = 4
n_hidden = 4
n_outputs = 1
initializer = tf.contrib.layers.variance_scaling_initializer()
learning_rate = 0.01

X = tf.placeholder(tf.float32,shape=[None,n_inputs])
hidden = tf.layers.dense(X,n_hidden,activation=tf.nn.elu,
                                kernel_initializer=initializer)
logits = tf.layers.dense(hidden,n_outputs,kernel_initializer=initializer)
outputs = tf.nn.sigmoid(logits)
# Selection of a random action based on estimated probabilities
p_left_and_right = tf.concat(axis=1,values=[outputs,1-outputs])
action = tf.multinomial(tf.log(p_left_and_right),num_samples=1)
# 
y = 1. - tf.to_float(action)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad,variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []
for grad,variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32,shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder,variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

def discount_rewards(rewards,discount_rate):
    ''' Calculate dicount rewards based on regular rewards. '''
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] =  cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards,discount_rate):
    ''' Normalize discounted rewards '''
    all_discounted_rewards = [discount_rewards(rewards,discount_rate)
                                            for rewards in all_rewards]
    # Straightens a vector into as (-1,1)
    flat_rewards = np.concatenate(all_discounted_rewards)
    # Mean value
    reward_mean = flat_rewards.mean()
    # Computes a standard deviation
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std
            for discounted_rewards in all_discounted_rewards]

a = discount_rewards([10,0,-50],discount_rate=0.8)
print(a)
b = discount_and_normalize_rewards([[10,0,-50],[10,20]],discount_rate=0.8)
print(b)

# Starting to train our policy:
# Number of iterations
n_iterations = 500
# Number of steps in one episode
n_max_steps = 1000
# Training out policy every 10 episodes
n_games_per_update = 10
# Saving out model every 10 iter 92 n_iterations = 250
save_iterations = 10
discount_rate = 0.95
'''
with tf.Session() as sess:
    init.run()
    i = 0
    for iteration in range(n_iterations):
        i += 1
        all_rewards = []
        all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                action_val,gradients_val = sess.run(
                                [action,gradients],
                                feed_dict={X:obs.reshape(1,n_inputs)})
                obs,reward,done,info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)
        all_rewards = discount_and_normalize_rewards(all_rewards,discount_rate)
        feed_dict = {}
        for var_index,grad_placeholder in enumerate(gradient_placeholders):
            # Multiplying gradients with rewards
            mean_gradients = np.mean(
                    [reward*all_gradients[game_index][step][var_index]
                        for game_index,rewards in enumerate(all_rewards)
                        for step,reward in enumerate(rewards)],axis=0)
            feed_dict[grad_placeholder] = mean_gradients
        sess.run(training_op,feed_dict=feed_dict)
        if iteration%save_iterations == 0:
            saver.save(sess,"./my_policy_net_pg.ckpt")
            print('{} iteration is done, saving results!'.format(i))
'''

with tf.Session() as sess:
    obs = env.reset()
    for i in range(10000):
        saver.restore(sess,"./my_policy_net_pg.ckpt")
        action_eval = action.eval(feed_dict={X:obs.reshape(1,n_inputs)})
        obs,reward,done,info = env.step(action_eval[0][0])
        if done:
            break
        env.render()
        


