import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
env = gym.make("CartPole-v0").env
env.reset()
n_actions = env.action_space.n
state_dim = env.observation_space.shape

plt.imshow(env.render("rgb_array"))
#building the neural network
import tensorflow as tf
import keras
import keras.layers as L
tf.reset_default_graph()
sess = tf.InteractiveSession()
keras.backend.set_session(sess)

network = keras.models.Sequential()
network.add(L.InputLayer(state_dim))
hidden_1=30
hidden_2=15
# let's create a network for approximate q-learning following guidelines above
network.add(L.Dense(hidden_1))
network.add(L.Dense(hidden_2))
#insert final layer
network.add(L.Dense(n_actions, activation='linear'))


def get_action(state, epsilon=0):
    """
    sample actions with epsilon-greedy policy
    recap: with p = epsilon pick random action, else pick action with highest Q(s,a)
    """

    q_values = network.predict(state[None])[0]
    random_num=np.random.rand()


    if random_num < epsilon:
        action=env.action_space.sample()
    else:
        action= np.argmax( q_values, axis=0)
    return action

assert network.output_shape == (None, n_actions), "please make sure your model maps state s -> [Q(s,a0), ..., Q(s, a_last)]"
assert network.layers[-1].activation == keras.activations.linear, "please make sure you predict q-values without nonlinearity"
print('Passed Network Tests ' )
# test epsilon-greedy exploration
s = env.reset()
assert np.shape(get_action(s)) == (), "please return just one action (integer)"
for eps in [0., 0.1, 0.5, 1.0]:
    state_frequencies = np.bincount([get_action(s, epsilon=eps) for i in range(10000)], minlength=n_actions)
    best_action = state_frequencies.argmax()
    assert abs(state_frequencies[best_action] - 10000 * (1 - eps + eps / n_actions)) < 200
    for other_action in range(n_actions):
        if other_action != best_action:
            assert abs(state_frequencies[other_action] - 10000 * (eps / n_actions)) < 200
    print('e=%.1f tests passed'%eps)

states_ph = tf.placeholder('float32'  , shape=(None,) + state_dim)
actions_ph = tf.placeholder('int32'  , shape=[None])
rewards_ph = tf.placeholder('float32'  , shape=[None])
next_states_ph = tf.placeholder('float32'  , shape=(None,) + state_dim)
is_done_ph = tf.placeholder('bool'  , shape=[None])

predicted_qvalues = network(states_ph)

#select q-values for chosen actions
predicted_qvalues_for_actions = tf.reduce_sum(predicted_qvalues * tf.one_hot(actions_ph, n_actions), axis=1)


gamma = 0.99

predicted_next_qvalues = network(next_states_ph)
# compute q-values for all actions in next states
predicted_qvalues_for_actions =tf.reduce_sum(predicted_next_qvalues * tf.one_hot(actions_ph, n_actions), axis=1)

# compute V*(next_states) using predicted next q-values
#return max q_val over actions
next_state_values = tf.reduce_max(predicted_next_qvalues, axis=0)


# compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
target_qvalues_for_actions =5 #change

# at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
#target_qvalues_for_actions = tf.where(is_done_ph, rewards_ph, target_qvalues_for_actions)

#mean squared error loss to minimize
loss = (predicted_qvalues_for_actions - tf.stop_gradient(target_qvalues_for_actions)) ** 2
loss = tf.reduce_mean(loss)

# training function that resembles agent.update(state, action, reward, next_state) from tabular agent
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

assert tf.gradients(loss, [predicted_qvalues_for_actions])[0] is not None, "make sure you update q-values for chosen actions and not just all actions"
assert tf.gradients(loss, [predicted_next_qvalues])[0] is None, "make sure you don't propagate gradient w.r.t. Q_(s',a')"
assert predicted_next_qvalues.shape.ndims == 2, "make sure you predicted q-values for all actions in next state"
assert next_state_values.shape.ndims == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
assert target_qvalues_for_actions.shape.ndims == 1, "there's something wrong with target q-values, they must be a vector"

def generate_session(t_max=1000, epsilon=0, train=False):
    """play env with approximate q-learning agent and train it at the same time"""
    total_reward = 0
    s = env.reset()

    for t in range(t_max):
        a = get_action(s, epsilon=epsilon)
        next_s, r, done, _ = env.step(a)

        if train:
            sess.run(train_step,{
                states_ph: [s], actions_ph: [a], rewards_ph: [r],
                next_states_ph: [next_s], is_done_ph: [done]
            })

        total_reward += r
        s = next_s
        if done: break

    return total_reward

epsilon = 0.5

for i in range(1000):
    session_rewards = [generate_session(epsilon=epsilon, train=True) for _ in range(100)]
    print("epoch #{}\tmean reward = {:.3f}\tepsilon = {:.3f}".format(i, np.mean(session_rewards), epsilon))

    epsilon *= 0.99
    assert epsilon >= 1e-4, "Make sure epsilon is always nonzero during training"

    if np.mean(session_rewards) > 300:
        print ("You Win!")
        break

#record sessions
import gym.wrappers
env = gym.wrappers.Monitor(gym.make("CartPole-v0"),directory="videos",force=True)
sessions = [generate_session(epsilon=0, train=False) for _ in range(100)]
env.close()

