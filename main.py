import gym
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random, choice


#
# def epsilon_greedy(Qtable, epsilon, current_state):
#     actions_values = Qtable[current_state, :]
#     max_value = max(actions_values)
#     greedy_actions = [act for act in range(len(actions_values)) if actions_values[act] == max_value]
#     explore = (random() < epsilon)
#     if explore:
#         return choice([act for act in range(len(actions_values))])
#     else:
#         return choice([act for act in range(len(greedy_actions))])


def epsilon_greedy(Q, epsilon, state):
    values = Q[state, :]
    max_value = max(values)
    num_of_actions = len(values)
    greedy_actions = [act for act in range(num_of_actions) if values[act] == max_value]
    explore = (random() < epsilon)
    if explore:
        return choice([act for act in range(num_of_actions)])
    else:
        return choice([act for act in greedy_actions])


def simulate(Q, epsilon,env):
    sum = 0.0
    for sim in range(15):
        current_state = env.reset()
        # env.render()
        done = False
        while not done:
            values = Q[env.env.s, :]
            max_value = max(values)
            no_actions = len(values)
            # action = epsilon_greedy(Q, epsilon, current_state)
            # state, reward, done, info = env.step(action)
            greedy_actions1 = [a for a in range(no_actions) if values[a] == max_value]
            state, reward, done, info = env.step(greedy_actions1[0])
            sum += reward
            # env.render()
    return (sum / 15)


def learn_sarsa(alpha, _lambda):
    takeSample = [250, 500, 750, 1000, 1500, 2000, 000, 5000, 7000, 10000, 13000, 16000, 20000]
    # env = gym.make("FrozenLake-v0")
    env = gym.make("FrozenLake8x8-v0")
    number_of_actions = env.action_space.n
    number_of_states = env.observation_space.n
    Q = np.zeros((number_of_states, number_of_actions))
    epsilon = 1.0
    gamma = 0.95
    # _lambda = 0.3
    # alpha = 0.1
    num_of_episodes = 20001
    # graph = []
    policy_value = []
    for episode in range(num_of_episodes):
        print(episode)
        if episode in takeSample:
            policy_value.append(simulate(Q, epsilon,env))
        # episode_reward = 1.0
        E = np.zeros((number_of_states, number_of_actions))
        current_state = env.reset()
        current_action = epsilon_greedy(Q, epsilon, current_state)
        done = False
        step_number = 0
        while not done and step_number < 250:
            next_state, reward, done, info = env.step(current_action)
            # episode_reward+=reward
            step_number += 1
            next_action = epsilon_greedy(Q, epsilon, next_state)
            delta = reward + gamma * Q[next_state, next_action] - Q[current_state, current_action]
            E[current_state, current_action] += 1
            # for i in range(number_of_states):
            #     for j in range(number_of_actions):
            #         Q[i, j] = Q[i, j] + alpha * delta * E[i, j]
            #         E[i, j] = _lambda * gamma * E[i, j]
            Q = Q + alpha * delta * E
            E = _lambda * gamma * E
            current_state = next_state
            current_action = next_action
        epsilon = epsilon * 0.999
        # graph.append(episode_reward)





    env.reset()
    env.render()
    done = False
    while not done:
        values = Q[env.env.s, :]
        max_value = max(values)
        no_actions = len(values)
        greedy_actions1 = [a for a in range(no_actions) if values[a] == max_value]
        state, reward, done, info = env.step(greedy_actions1[0])
        # action = epsilon_greedy(Q, epsilon, current_state)
        # state, reward, done, info = env.step(action)
        env.render()
    env.reset()


    plt.plot(policy_value, linewidth=2)
    plt.xlabel(f"Episode   alpha= {alpha}  lambda= {_lambda} ")
    plt.ylabel("episode value".format(100))
    # plt.axis([0, 20000, 0, 1])
    plt.show()



if __name__ == '__main__':
  learn_sarsa(0.05,0.3)
  learn_sarsa(0.1,0.3)
  learn_sarsa(0.1,0.2)
  learn_sarsa(0.05,0.2)

