import tensorflow as tf
import joblib
import numpy as np
import os
import multiprocessing
from gym.spaces import Discrete, Box

def make_session(config=None,num_cpu=None,make_default=False,graph=None):
    if num_cpu is None:
        num_cpu=int(os.getenv("RCALL_NUM_CPU",multiprocessing.cpu_count()))
    if config is None:
        config=tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu,
            )
        config.gpu_options.allow_growth=True
    if make_default:
        return tf.InteractiveSession(config=config,graph=graph)
    else:
        return tf.Session(config=config,graph=graph)



def get_session(config):
    sess=tf.get_default_session()
    if sess is None:
        sess=make_session(config=config,make_default=True)
    return sess

def orth_init_(scale=1.0):
    def _orth_init(shape,dtype,partition_info=None):
        shape=tuple(shape)
        if len(shape)==2:
            flat_shape=shape
        elif len(shape)==4:
            flat_shape=(np.prod(shape[:-1]),shape[-1])
        else:
            raise NotImplemented
        a=np.random.normal(0.0,1.0,flat_shape)
        u,_,v=np.linalg.svd(a,full_matrices=False)
        q=u if u.shape==flat_shape else v
        q=q.reshape(shape)
        return (scale*q[:shape[0],:shape[1]]).astype(np.float32)


def fc(x,scope,nh,init_scale=1.0,init_bias=0.0):
    with tf.variable_scope(scope):
        nin=x.get_shape()[1].value
        w=tf.get_variable("w",[nin,nh],initializer=ortho_init(init_scale))
        b=tf.get_variable("b",[nh],initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x,w)+b

def observation_placeholder(ob_space,batch_size=None,name="Ob"):
    dtype=ob_space.dtype
    if dtype==np.int8:
        dtype=np.uint8
    return tf.placeholder(shape=(batch_size,)+ob_space.shape,dtype=dtype,name=name)

def encode_observation(ob_space,place_holder):
    if isinstance(ob_space,Discrete):
        return tf.to_float(tf.one_hot(place_holder),ob_space.n)
    elif isinstance(ob_space,Box):
        return tf.to_float(place_holder)
    else:
        raise NotImplementedError



def save_state(fname, sess=None):
    from baselines import logger
    logger.warn('save_state method is deprecated, please use save_variables instead')
    sess = sess or get_session()
    dirname = os.path.dirname(fname)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    saver = tf.train.Saver()
    saver.save(tf.get_default_session(), fname)

def adjust_shape(data,placeholder):
    if not isinstance(data,np.ndarray) and not isinstance(data,list):
        return data
    if isinstance(data,list):
        data=np.array(data)
    placeholder_shape=[x or -1 for x in placeholder.shape.as_list()]

    return np.reshape(data,placeholder_shape)


def save_variables(save_path, variables=None, sess=None):
    sess = sess or get_session()
    variables = variables or tf.trainable_variables()

    ps = sess.run(variables)
    save_dict = {v.name: value for v, value in zip(variables, ps)}
    dirname = os.path.dirname(save_path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(save_dict, save_path)

def load_variables(load_path, variables=None, sess=None):
    sess = sess or get_session()
    variables = variables or tf.trainable_variables()

    loaded_params = joblib.load(os.path.expanduser(load_path))
    restores = []
    if isinstance(loaded_params, list):
        assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
        for d, v in zip(loaded_params, variables):
            restores.append(v.assign(d))
    else:
        for v in variables:
            restores.append(v.assign(loaded_params[v.name]))

    sess.run(restores)