import gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv,VecNormalize

from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines import PPO2


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









# multiprocess environment
n_cpu = 1
env = DummyVecEnv([EnvFunc for i in range(n_cpu)])
env = VecNormalize(env,norm_obs=True, norm_reward=False)

model = PPO2(MlpPolicy, env, n_steps=256,nminibatches=8,verbose=1)
model.learn(total_timesteps=100000000)
model.save("ppo2_cartpole")