import numpy as np
import pandas as pd
import gym


n_bins = 4
Q = np.zeros([n_bins**4,2])
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

def concatInfo(p,s,a,r):
    return (str(p)+str(s)+str(a)+str(r))

def getIndex(n3,n2,n1,n0):
    # se le pasan las 4 observaciones, y se calcula la posicion en la Q-tabla
    return 64*n3+16*n2+4*n1+n0

alpha = 0.5
max_episodios = 200

def Qlearn(episodes):
    env = gym.make('CartPole-v1')
    for e in range(episodes):
        done = False
        p,q,s,r = toDiscrete(env.reset())

        state = getIndex(p,q,s,r)
        counter = 0
        R = 0
        i = 0
        while i < (max_episodios): #200 como máximo de pasos a dar
            #c = contar(Q)
            counter += 1
            #for r in range(1): env.render()

            #print("episodio = ",e," paso = ",i)
            #env.render() #si la recompensa es peor que el que hay no sobreescibir el estado, o si?
            #v0,v1 = Q[state][0],Q[state][1]
            a = np.argmax(Q[state])
            action = a
            observation, reward, done, info = env.step(action)
            p,s,q,r = toDiscrete(observation)
            state2 = getIndex(p,q,s,r)
            if done:
                reward = -10

            #if (done and (i<(max_episodios-10))):
             #   i = max_episodios-max
            i+=1
            Q_valor = Q[state, action] + alpha * (reward + np.max(Q[state2]) - Q[state, action])


            Q[state, action] = Q_valor
            state = state2
            R +=reward

        if e %500 == 0:
            print("Episodio {}, recompensa -> {}".format(e,R))

        #print(observation)
        #print("Contador =",counter)
        #print("Reward = ",R)


def contar(Q):
    cont = 0
    n0,n1 = 0,0
    for i in range(len(Q)):
        if Q[i][0]!=0:
            n0+=1
        if Q[i][1]!=0:
            n1+=1
    return n0,n1

episodios = 5000
Qlearn(episodios)
n0,n1 = contar(Q)
print("n0 = ",n1,"\nn1 = ",n1)
env = gym.make('CartPole-v1')
p,q,s,r = toDiscrete(env.reset())
#print(Q)
print("El mayor valor es:",np.amax(Q))
state = getIndex(p,q,s,r)
for i in range(100):
    env.render()
    action = np.argmax(Q[state])
    observation, reward, done, info = env.step(action)
    p, s, q, r = toDiscrete(observation)
    state = getIndex(p,s,q,r)









