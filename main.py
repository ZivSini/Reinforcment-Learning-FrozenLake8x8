import gym
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random, choice


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


def simulate(Q,env):
    sum = 0.0
    for sim in range(100):
        current_state = env.reset()
        done = False
        while not done:
            values = Q[env.env.s, :]
            max_value = max(values)
            no_actions = len(values)
            greedy_actions1 = [a for a in range(no_actions) if values[a] == max_value]
            state, reward, done, info = env.step(greedy_actions1[0])
            sum += reward
    return (sum / 100)


def learn_sarsa(alpha, _lambda):
    takeSample=np.linspace(1000,30000,30)
    takeSample=np.concatenate((takeSample,np.linspace(30000,80000,25)))
    takeSample=np.concatenate((takeSample,np.linspace(80000,200000,40)))
    takeSample=np.concatenate((takeSample,np.linspace(200000,500000,60)))
    takeSample=np.concatenate((takeSample,np.linspace(500000,1000000,50)))
    takeSample=np.concatenate((takeSample,np.linspace(1000000,2000000,50)))


    # env = gym.make("FrozenLake-v0")
    env = gym.make("FrozenLake8x8-v0")
    number_of_actions = env.action_space.n
    number_of_states = env.observation_space.n
    Q = np.zeros((number_of_states, number_of_actions))
    epsilon = 1.0
    gamma = 0.95
    policy_value = []
    counter = 0
    next_sample=0
    while counter<1000000:
        if next_sample<len(takeSample) and counter>takeSample[next_sample]:
            policy_value.append(simulate(Q,env))
            next_sample+=1
        E = np.zeros((number_of_states, number_of_actions))
        current_state = env.reset()
        current_action = epsilon_greedy(Q, epsilon, current_state)
        done = False
        step_number = 0
        while not done and step_number < 300:
            counter += 1
            next_state, reward, done, info = env.step(current_action)
            step_number += 1
            next_action = epsilon_greedy(Q, epsilon, next_state)
            delta = reward + gamma * Q[next_state, next_action] - Q[current_state, current_action]
            E[current_state, current_action] += 1
            Q = Q + alpha * delta * E
            E = _lambda * gamma * E
            current_state = next_state
            current_action = next_action
        epsilon = epsilon * 0.999
    env.reset()
    env.render()
    done = False
    while not done:
        values = Q[env.env.s, :]
        max_value = max(values)
        no_actions = len(values)
        greedy_actions1 = [a for a in range(no_actions) if values[a] == max_value]
        state, reward, done, info = env.step(greedy_actions1[0])
        env.render()
    env.reset()


    plt.plot(takeSample[:len(policy_value)],policy_value, linewidth=2)
    plt.xlabel(f"time steps\n   alpha= {alpha}  lambda= {_lambda} ")
    plt.ylabel("episode value".format(100))
    plt.show()



if __name__ == '__main__':
  print("running...")
  learn_sarsa(0.15, 0.7)
  learn_sarsa(0.15, 0.3)
  learn_sarsa(0.05, 0.7)
  learn_sarsa(0.05, 0.3)

  # learn_sarsa(0.15, 0.5)
  #
  # learn_sarsa(0.1, 0.2)
  # learn_sarsa(0.2, 0.2)
  # learn_sarsa(0.1, 0.4)
  # learn_sarsa(0.2, 0.4)
  # #
  # learn_sarsa(0.2, 0.4)
  # learn_sarsa(0.2, 0.4)
  # learn_sarsa(0.2, 0.4)
  # learn_sarsa(0.2, 0.4)



