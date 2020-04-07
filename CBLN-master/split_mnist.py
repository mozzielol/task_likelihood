import numpy as np
from bnn.model_utils import *
from bnn.BNN_MLP import *
from bnn.utils import *
from bnn.train_utils import *

'''
Tested with: 
	- tensorflow : 1.13.1
	- dm-sonnet : 1.29
	- tensorflow-probability : 0.6.0
	- numpy 
	- scipy
	- keras
	- tqdm
'''
"""
The model may fail to identify the test data, we allow to search for the best model which can distinguish the test data.
"""


num_epochs = 5
num_neurons = 10

def train(task_labels):
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
	    NUM_TASK=len(task_labels)
	    net = BNN_MLP(n_inputs=784, n_outputs=10, hidden_units=[10,10],num_task=NUM_TASK, init_mu=0.0, init_rho=-3.0, 
              activation=tf.nn.relu)
	    train_task = construct_split_mnist(task_labels)
	    test_task = construct_split_mnist(task_labels,split='test')
	    train_init,test_init = load_iterator(net,train_task,test_task)
	    sess.run( tf.global_variables_initializer() )
	    em_train(net,sess,num_epochs,train_task,test_task,train_init,test_init,lams=[0.01],search_best=True)
    
    

task_labels = [[0,1,2,3],[4,5,6],[7,8,9]]
train(task_labels)



