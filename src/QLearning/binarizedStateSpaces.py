import gym
import matplotlib.pyplot as plt
import numpy as np
env = gym.make("CartPole-v0")
n_actions = env.action_space.n

print("first state:%s" % (env.reset()))
plt.imshow(env.render('rgb_array'))
all_states = []
for _ in range(1000):
    all_states.append(env.reset())
    done = False
    while not done:
        s, r, done, _ = env.step(env.action_space.sample())
        all_states.append(s)
        if done: break

all_states = np.array(all_states)

for obs_i in range(env.observation_space.shape[0]):
    plt.hist(all_states[:, obs_i], bins=20)
    plt.show()
from gym.core import ObservationWrapper
class Binarizer(ObservationWrapper):

    def _observation(self, state):
        print(state)
        input()
        #state = <round state to some amount digits.>
        #hint: you can do that with round(x,n_digits)
        #you will need to pick a different n_digits for each dimension

        return tuple(state)
env = Binarizer(gym.make("CartPole-v0"))
all_states = []
for _ in range(1000):
    all_states.append(env.reset())
    done = False
    while not done:
        s, r, done, _ = env.step(env.action_space.sample())
        all_states.append(s)
        if done: break

all_states = np.array(all_states)

for obs_i in range(env.observation_space.shape[0]):

    plt.hist(all_states[:,obs_i],bins=20)
    plt.show()
agent = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99,
                       getLegalActions = lambda s: range(n_actions))
rewards = []
for i in range(1000):
    rewards.append(play_and_train(env,agent))

    #OPTIONAL YOUR CODE: adjust epsilon
    if i %100 ==0:
        clear_output(True)
        print('eps =', agent.epsilon, 'mean reward =', np.mean(rewards[-10:]))
        plt.plot(rewards)
        plt.show()

