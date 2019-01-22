import gym
import numpy as np
import ppo

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
class MyReward(gym.Wrapper):
    def __init__(self, env):
        super(MyReward, self).__init__(env)
        self.m_RwardList = []
        self.m_count = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # reward=0
        self.m_count += 1
        if self.m_count%60000==0:
            print("frame",self.m_count)
        # print("reware",self.m_count,action,reward,done,info)
        if not done:
            self.m_RwardList.append(reward)
        else:
            # if info["done"]:
            #     reward=100
            #     self.m_RwardList.append(reward)
            iMeanReward = np.sum(self.m_RwardList)
            print("mean_reward", iMeanReward)
            self.m_RwardList = []
        return obs, reward, done, info


def EnvFunc():
    oEnv = gym.make("MountainCar-v0")
    oEnv = MyReward(oEnv)
    return oEnv

def Train():
    env=DummyVecEnv([EnvFunc]*1)
    env = VecNormalize(env,ob=True, ret=True)
    act=ppo.learn(
        network="mlp",
        env=env,
        # lr=3e-4,
        nsteps=256,
        nminibatches=8,
        # lam=0.94,
        total_timesteps=6000000,
        # log_interval=100,
        # save_interval=500,
        num_layers=3,
        num_hidden=256,
        value_network="copy"
    )



if __name__=="__main__":
    Train()