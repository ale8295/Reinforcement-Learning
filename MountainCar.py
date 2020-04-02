import gym
import numpy as np
import pandas as pd
import random

env = gym.make('MountainCar-v0')

# n_bins -> En valores queremos discretizar los valores continuos
# observation_space.n -> tamaño de cada observacion
# action_space -> número de acciones distintas
# creamos una Q-tabla de tamaño [b_bins**observation_space,action_space]
# [4**2,3] -> [16,3]

n_bins = 20
Q = np.zeros([n_bins**2,env.action_space.n])

position_bins = pd.cut([-1.2,0.6], bins=n_bins, retbins=True)[1][1:-1]
speed_bins = pd.cut([-0.07,0.07], bins=n_bins, retbins=True)[1][1:-1]

def toDiscrete(observation):
    pos_obs, speed_obs = observation
    pos = np.digitize(x=[pos_obs], bins=position_bins)[0]
    speed = np.digitize(x=[speed_obs], bins=speed_bins)[0]
    return getIndex(pos,speed)

def concatInfo(p,s,a,r):
    return (str(p)+str(s)+str(a)+str(r))

def getIndex(n1,n0):
    # se le pasan las 4 observaciones, y se calcula la posicion en la Q-tabla
    return (env.action_space.n)*n1+n0

def transformarReward(p):
    rt = 0
    if p<0:
        rt = p
    else:
        rt = p*100

    return rt

episodios = 1000
cambiar = 0.2
alpha = 0.9
for i in range(episodios):

    observation = env.reset()
    done = False
    while not done:
        r = random.random()
        #if i > (episodios -100):
        env.render()
        state = toDiscrete(observation)
        pos, speed = observation
        if r < cambiar:
            action = random.randint(0,2)
        else:
            action = np.argmax(Q[state])
        observation, reward, done, info = env.step(action)
        reward = reward+transformarReward(pos)
        state2 = toDiscrete(observation)
        Q_valor = Q[state, action] + alpha * (reward + np.max(Q[state2]) - Q[state, action])
        Q[state, action] = Q_valor
        state = state2


print(Q)