import gym
env = gym.make("Taxi-v2")
print("Creating environment space" )
n_actions = env.action_space.n
from simpleQLearning import QLearningAgent
print("Creating Q Learning Agent" )
agent = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99,
                       get_legal_actions = lambda s: range(n_actions))
def play_and_train(env,agent,t_max=10**4):
    """
    This function should
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    s = env.reset()
    print("Beginning Training: " )
    for t in range(t_max):
        print('Training...',t )
        # get agent to pick action given state s.
        a = agent.get_action(s)

        next_s, r, done, _ = env.step(a)

        # train (update) agent for state s
        agent.update(s, a, r, next_s)
        s = next_s
        total_reward +=r
        if done: break

    return total_reward



rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    agent.epsilon *= 0.99

    if i %100 ==0:
        clear_output(True)
        print('eps =', agent.epsilon, 'mean reward =', np.mean(rewards[-10:]))
        plt.plot(rewards)
        plt.show()

