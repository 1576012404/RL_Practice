import tensorflow as tf
import numpy as np

pred_action=[[2,2.],[3,4],[4,5]]
label_action=[0,1,0]
n_action=2

all_act_prob=tf.nn.softmax(pred_action)

neg_log_prob=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_action,labels=label_action)


label_onehot=tf.one_hot(label_action,n_action)
nge_log_prob2=tf.reduce_sum(-tf.log(all_act_prob)*tf.one_hot(label_action,n_action),axis=1)

neg_log_prob3=tf.nn.softmax_cross_entropy_with_logits(logits=pred_action,labels=label_onehot)

with tf.Session() as sess:# alll the same
    print(sess.run(neg_log_prob))
    print(sess.run(nge_log_prob2))
    print(sess.run(neg_log_prob3))