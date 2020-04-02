import numpy as np
import gym
import pandas as pd
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
ev_reward = []
ev_pr = []
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 40000
SHOW_EVERY = 100001
MAX_EPISODES = 200
SHOW_INFO = 500

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//10
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

n_bins = 4
Q = np.random.uniform(low= 0,high = 10,size = [n_bins**4,2])
cart_position_bins = pd.cut([-4.8, 4.8], bins=n_bins, retbins=True)[1][1:-1]
cart_velocity_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
pole_angle_bins = pd.cut([-2, 2], bins=n_bins, retbins=True)[1][1:-1]
angle_rate_bins = pd.cut([-3.5, 3.5], bins=n_bins, retbins=True)[1][1:-1]

def toDiscrete(observation):
    cart_position, cart_velocity, pole_angle, angle_rate_of_change = observation
    cP = np.digitize(x=[cart_position], bins=cart_position_bins)[0]
    cS = np.digitize(x=[cart_velocity], bins=cart_position_bins)[0]
    pA = np.digitize(x=[pole_angle], bins=cart_position_bins)[0]
    aR = np.digitize(x=[angle_rate_of_change], bins=cart_position_bins)[0]
    return cP, cS, pA, aR

def getState(observation):
    # se le pasan las 4 observaciones, y se calcula la posicion en la Q-tabla
    n3,n2,n1,n0 = toDiscrete(observation)
    return 64*n3+16*n2+4*n1+n0


for episode in range(EPISODES):

    done = False
    state = getState(env.reset())
    R = 0
    i = 0
    while i < MAX_EPISODES:
        i += 1
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(Q[state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        observation, reward, done, _ = env.step(action)
        R+=reward
        new_state = getState(observation)

        #if episode % SHOW_EVERY == 0:
            #env.render()


        max_future_q = np.max(Q[new_state])

        # Current Q value (for current state and performed action)
        current_q = Q[state,action]

        # And here's our equation for a new Q value for current state and action
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        # Update Q table with new Q value
        Q[state,action] = new_q
        state = new_state

    ev_reward.append(R)
    if episode % SHOW_INFO == 0:
        print("Episodio ->", episode, "Reward =",R)
        ev_pr.append(R)

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()

plt.plot(ev_pr)
plt.show()

state = getState(env.reset())
print(state)
for i in range(MAX_EPISODES):

    action = np.argmax(Q[state])
    observation, reward, done, info = env.step(action)
    state = getState(observation)
    env.render()