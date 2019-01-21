from policy import build_policy
from runer import Runner

def learn(network,env,total_timesteps,nsteps=2048,
          lr=1e-4,gamma=0.99,lam=0.95,
          max_grad_norm=0.5,cliprange=0.2,
          ent_coef=0.0,vf_coef=0.5,
          nminibatches=4,noptepochs=4,
          model_fn=None,
          load_path=None,
          **netword_kargs):#value_network

    total_timesteps=int(total_timesteps)
    nenvs=env.num_envs
    ob_space=env.observation_space
    ac_space=env.action_space
    nbatch=nenvs*nsteps
    nbatch_train=nbatch//nminibatches

    if model_fn is None:
        from model import Model
        model_fn=Model

    policy=build_policy(env,network,**netword_kargs)
    model=model_fn(policy=policy,ob_space=ob_space,ac_space=ac_space,nbatch_act=nenvs,nbatch_train=nbatch_train,nsteps=nsteps,
                   ent_coef=ent_coef,vf_coef=vf_coef,max_grad_norm=max_grad_norm)
    runner=Runner(env=env,model=model,nsteps,gamma=gamma,lam=lam)

