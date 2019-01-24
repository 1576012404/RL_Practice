import gym
import time
#
#BreakoutNoFrameskip-v4
env_id = "PongNoFrameskip-v4"
env = gym.make(env_id)

env.reset()
gap=0
while True:
    action=env.action_space.sample()
    ob_,reward,done,info=env.step(action)
    # print("reward",reward)
    gap+=1
    if reward!=0:
        print("gap",gap)
        gap=0

    # print("info",reward,done)
    env.render()
    time.sleep(0.05)
    if done:
        obs=env.reset()

