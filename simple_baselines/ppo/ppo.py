from policy import build_policy
from runer import Runner
from simple_baselines.utils import constfn

import time
import numpy as np

def learn(network,env,total_timesteps,nsteps=2048,
          lr=1e-4,gamma=0.99,lam=0.95,
          max_grad_norm=0.5,cliprange=0.2,
          ent_coef=0.0,vf_coef=0.5,
          nminibatches=4,noptepochs=4,
          model_fn=None,
          load_path=None,
          log_interval=100,
          **netword_kargs):#value_network

    if isinstance(lr,float):lr=constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange,float):cliprange=constfn(cliprange)
    else:assert callable(cliprange)

    total_timesteps=int(total_timesteps)
    nenvs=env.num_envs
    ob_space=env.observation_space
    ac_space=env.action_space
    nbatch=nenvs*nsteps
    nbatch_train=nbatch//nminibatches
    assert  nbatch%nminibatches==0

    if model_fn is None:
        from model import Model
        model_fn=Model

    policy=build_policy(env,network,**netword_kargs)
    model=model_fn(policy=policy,ob_space=ob_space,ac_space=ac_space,nbatch_act=nenvs,nbatch_train=nbatch_train,nsteps=nsteps,
                   ent_coef=ent_coef,vf_coef=vf_coef,max_grad_norm=max_grad_norm)
    runner=Runner(env=env,model=model,nsteps=nsteps,gamma=gamma,lam=lam)

    nupdates=total_timesteps//nbatch
    tfirststart = time.time()

    for update in range(1,nupdates+1):
        tstart=time.time()
        frac=1.0-(update-1.0)/nupdates
        lrnow=lr(frac)
        cliprangenow=cliprange(frac)
        obs,returns,masks,actions,values,neglogps=runner.run()
        inds=np.arange(nbatch)
        mblossvals=[]
        for _ in range(noptepochs):
            np.random.shuffle(inds)
            for start in range(0,nbatch,nbatch_train):
                end=start+nbatch_train
                mbinds=inds[start:end]
                slices=( attr[mbinds] for attr in (obs,returns,masks,actions,values,neglogps))
                mblossvals.append(model.train(lrnow,cliprangenow,*slices))
        lossvals=np.mean(mblossvals,axis=0)
        tnow=time.time()
        fps=int(nbatch/(tnow-tstart))

        if update%log_interval==0:
            print("*******status********")
            print("total_timesteps:",update*nbatch)
            print("time_cost:",tnow-tfirststart)
            for (loss_name,loss_val) in zip(model.loss_name,lossvals):
                print("%s:"%loss_name,loss_val)
            print("********************")
    return model


