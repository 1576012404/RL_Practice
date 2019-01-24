import gym
import numpy as np
import gym
import numpy as np
from baselines.ddpg import ddpg


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
    oEnv = gym.make("MountainCarContinuous-v0")
    oEnv = MyReward(oEnv)
    return oEnv









# multiprocess environment
n_cpu = 2
env = DummyVecEnv([EnvFunc for i in range(n_cpu)])
env = VecNormalize(env,ob=True, ret=False)
print("ac_shape",env.action_space.shape)
print("ob_shape",env.observation_space.shape)

model = ddpg.learn("mlp",env=env,total_timesteps=int(1e6),)