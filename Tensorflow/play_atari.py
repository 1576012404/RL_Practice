import gym
import time

env_id = "BreakoutNoFrameskip-v4"
env = gym.make(env_id)

env.reset()
while True:
    action=env.action_space.sample()
    ob_,reward,done,info=env.step(action)
    print("info",reward,done)
    env.render()
    time.sleep(0.05)
    if done:
        obs=env.reset()

