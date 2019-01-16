import gym
import numpy as np
from baselines.ppo2 import ppo2

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from baselines.common.vec_env.vec_normalize import VecNormalize

from baselines.common.atari_wrappers import make_atari, wrap_deepmind

class MyReward(gym.Wrapper):
    def __init__(self, env):
        super(MyReward, self).__init__(env)
        self.m_RwardList = []
        self.m_Lost=[]
        self.m_count = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # reward=0
        self.m_count += 1
        # print("reware",self.m_count,action,reward,done,info)
        if reward>0:
            self.m_RwardList.append(reward)
        elif reward<0:
            self.m_Lost.append(reward)
        if done:
            # if info["done"]:
            #     reward=100
            #     self.m_RwardList.append(reward)
            iMeanReward = np.sum(self.m_RwardList)
            iLost = np.sum(self.m_Lost)
            print("mean_reward", iMeanReward,iLost)
            self.m_RwardList = []
            self.m_Lost = []
        return obs, reward, done, info


def EnvFunc():
    env_id = "PongNoFrameskip-v4"
    oEnv = make_atari(env_id)
    oEnv=wrap_deepmind(oEnv,frame_stack=True)
    oEnv=MyReward(oEnv)
    return oEnv

def Train():


    env=DummyVecEnv([EnvFunc]*8)
    # env = VecNormalize(env,ob=True, ret=False)
    act=ppo2.learn(
        network="cnn",
        env=env,
        # lr=3e-4,
        nsteps=128,
        nminibatches=4,
        # lam=0.94,
        total_timesteps=6000000,
        log_interval=80,
        # save_interval=500,
        # num_layers=3,
        # num_hidden=256,
        value_network="copy"
    )
    act.save("ppo2_model/model")



if __name__=="__main__":
    Train()