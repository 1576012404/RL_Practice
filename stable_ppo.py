import gym
import numpy as np

from stable_baselines import PPO2

from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise

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


def EnvFunc(env_id,n_envs,iSeed):
    env = make_atari_env(env_id, num_env=n_envs, seed=iSeed)
    return env


def linear_schedule(initial_value):
    def func(process):
        return process*initial_value

    return func


learning_rate=linear_schedule(2.5e-4)
clip_range=linear_schedule(0.1)
n_timesteps=10000000
hyperparmas ={'policy': 'CnnPolicy', 'n_steps': 128, 'noptepochs': 4, 'nminibatches': 4, 'learning_rate': learning_rate, 'cliprange':clip_range ,
            'vf_coef': 0.5, 'ent_coef': 0.01}
def Train():
    env_id = "PongNoFrameskip-v4"
    env=EnvFunc(env_id,8,0)
    model = PPO2(env=env, verbose=1, **hyperparmas)

    model.learn(n_timesteps)



if __name__=="__main__":
    Train()