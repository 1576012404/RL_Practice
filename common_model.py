import tensorflow as tf
import numpy as np
from utils import fc

mapping={}

def register(name):
    def _thunk(func):
        mapping[name]=func
        return func
    return _thunk


@register("mlp")
def mlp(num_layers=2,num_hidden=64,activation=tf.tanh,layer_norm=False):
    def network_fun(X):
        h=tf.layers.flatten(X)
        for i in range(num_layers):
            h=fc(h,"mlp_fc{}".format(i),nh=num_hidden,init_scale=np.sqrt(2))
            if layer_norm:
                h=tf.contrib.layers.layer_norm(h,center=True,scale=True)
            h=activation(h)
        return h
    return network_fun




def get_network_builder(name):

    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError("Unknow network type:{}".format(name))


# if __name__ =="__main__":
#     a=get_network_builder("mlp")
#     print(a)