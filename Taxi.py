import gym
import numpy as np

env = gym.make('Taxi-v3')
state = env.reset()

episodios = 50
counter = 0
reward = None
Q = np.zeros([env.observation_space.n,env.action_space.n])
G = 0
print(Q)
alpha = 0.618
counter = 0
pasos = []
for episode in range(1,1001):
    done = False
    G,reward = 0,0
    state = env.reset()
    while done != True:

        action = np.argmax(Q[state])
        state2, reward, done, info = env.step(action)
        Q[state,action] += alpha * (reward + np.max(Q[state2])- Q[state,action])
        G += reward
        state = state2



    if episode % 500 == 0:
        print('Episode {} Total Reward: {}'.format(episode,G))

print(Q[3][1])


state = env.reset()
env.render()
reward = None
counter = 0
while reward != 20:
    env.render()
    action = np.argmax(Q[state])
    state, reward, done, info = env.step(action)
    counter +=1

print("Resulto en {} pasos".format(counter))
print(Q)
print(np.argmax(Q[1]))