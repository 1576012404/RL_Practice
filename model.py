import tensorflow as tf
import functools
from utils import get_session,save_variables, load_variables

class Model():
    def __init__(self,policy,ob_space,ac_space,nbatch_act,nbatch_train,nsteps,ent_coef,vf_coef,max_grad_norm):
        self.sess=sess=get_session()

        with tf.variable_scope("ppo_model",reuse=tf.AUTO_REUSE):
            act_model=policy(nbatch_act,1,sess)
            train_model=policy(nbatch_train,nsteps,sess)
        self.A=A=train_model.pdtype.sample_shape([None])
        self.ADV=ADV=tf.placeholder(tf.float32,[None])
        self.R=R=tf.placeholder(tf.float32,[None])

        self.OLDNEGLOGP=OLDNEGLOGP=tf.palceholder(tf.float32,[None])
        self.OLDVPRED=OLDVPRED=tf.placeholder(tf.float32,[])
        self.LR=LR=tf.placeholder(tf.float32,[])

        self.CLIPRANGE=CLIPRANGE=tf.placeholder(tf.float32,[])

        neglogp=train_model.pd.neglogp(A)
        entropy=tf.reduce_mean(train_model.pd.entropy())

        vpred=train_model.vf
        vpredclipped=OLDVPRED+tf.clip_by_value(train_model.vf-OLDVPRED,-CLIPRANGE,CLIPRANGE)
        vf_losses1=tf.square(vpred-R)
        vf_losses2=tf.square(vpredclipped-R)

        vf_loss=0.5*tf.reduce_mean(tf.maxmum(vf_losses1,vf_losses2))

        ratio=tf.exp(OLDNEGLOGP-neglogp)
        pg_losses=-ADV*ratio
        pg_losses2=-ADV*tf.clip_by_value(ratio,1.0-CLIPRANGE,1.0+CLIPRANGE)

        pg_loss=tf.reduce_mean(tf.maximum(pg_losses,pg_losses2))

        approxkl=0.5*tf.reduce_mean(tf.square(neglogp-OLDNEGLOGP))
        clipfrac=tf.reducemean(tf.to_float(tf.greater(tf.abs(ratio-1),CLIPRANGE)))

        loss=pg_loss-entropy*ent_coef+vf_loss*vf_coef

        params=tf.trainable_variables("ppo2_model")
        self.trainer=tf.train.AdamOptimizer(learning_rate=LR,epsilon=1e-5)

        grads_and_var=self.trainer.compute_gradients(loss,params)
        grads,var=zip(*grads_and_var)

        if max_grad_norm is not None:
            grads,_grad_norm=tf.clip_by_global_norm(grads,max_grad_norm)
        grads_and_var=list(zip(grads,var))

        self.grads=grads
        self.var=var
        self._train_op=self.trainer.apply_gradients(grads_and_var)
        self.loss_name=["policy_loss","value_loss","policy_entropy","approxkl","clipfrac"]
        self.stats_list=[pg_loss,vf_loss,entropy,approxkl,clipfrac]

        self.train_model=train_model
        self.act_model=act_model
        self.step=act_model.step
        self.value=act_model.value

        self.save=functools.partial(save_variables,sess=sess)
        self.load=functools.partial(load_variables,sess=sess)

    def train(self,lr,cliprange,obs,returns,masks,actions,values,neglogps):

        advs=returns-values

        advs=(advs-advs.mean())/(advs.std()+1e-8)

        td_map={
            self.train_model.X:obs,
            self.A: actions,
            self.ADV : advs,
            self.R:returns,
            self.LR:lr,
            self.CLIPRANGE:cliprange,
            self.OLDNEGLOGP:neglogps,
            self.OLDVPRED:values
        }
        return self.sess.run(
            self.stats_list+[self._train_op],td_map
        )[:-1]








