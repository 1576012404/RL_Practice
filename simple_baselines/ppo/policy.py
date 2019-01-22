import tensorflow as tf

from baselines.common.distributions import make_pdtype
from simple_baselines.common_model import get_network_builder
from simple_baselines.utils import observation_placeholder,encode_observation,fc,adjust_shape

class PolicyWithValue():

    def __init__(self,env,observations,latent,vf_latent=None,sess=None):
        self.X=observations
        vf_latent=vf_latent if vf_latent is not None else latent
        latent = tf.layers.flatten(latent)
        vf_latent=tf.layers.flatten(vf_latent)
        self.pdtype=make_pdtype(env.action_space)
        self.pd,self.pi=self.pdtype.pdfromlatent(latent,init_scale=0.01)
        self.action=self.pd.sample()
        self.neglogp=self.pd.neglogp(self.action)

        self.sess=sess or tf.get_default_session()

        self.vf=fc(vf_latent,"vf",1)
        print("self.vf",self.vf.shape)
        self.vf=self.vf[:,0]
        print("vf2",self.vf.shape)

    def _evaluate(self,variable,observation,**extra_feed):
        sess=self.sess
        feed_dict={self.X:adjust_shape(observation,self.X)}
        return sess.run(variable,feed_dict)

    def step(self,observation,**extra_feed):
        a,v,neglogp=self._evaluate([self.action,self.vf,self.neglogp],observation)
        return a,v,neglogp

    def value(self,ob):
        return self._evaluate(self.vf,ob)






def build_policy(env,policy_network,value_network=None,**policy_kargs):
    if isinstance(policy_network,str):
        network_type=policy_network
        policy_network=get_network_builder(network_type)(**policy_kargs)

    def policy_fn(nbatch=None,nstep=None,sess=None,obs_placeholder=None):
        ob_space=env.observation_space
        X=obs_placeholder if obs_placeholder !=None else observation_placeholder(ob_space,batch_size=nbatch)
        X=encode_observation(ob_space,X)

        with tf.variable_scope("pi",reuse=tf.AUTO_REUSE):
            policy_latent=policy_network(X)

        v_net=value_network

        if v_net==None or v_net=="shared":
            vf_latent=policy_latent
        else:
            if value_network=="copy":
                v_net=policy_network
            else:
                assert  callable(v_net)
            with tf.variable_scope("vf",reuse=tf.AUTO_REUSE):
                vf_latent=v_net(X)
        policy=PolicyWithValue(
            env=env,
            observations=X,
            latent=policy_latent,
            vf_latent=vf_latent,
            sess=sess,
                )
        return policy
    return policy_fn
