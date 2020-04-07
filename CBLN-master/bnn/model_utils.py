import tensorflow as tf
import numpy as np
from tqdm import tqdm

def make_iterator(dataset,BATCH_SIZE):
    with tf.name_scope('data'):
        data = tf.data.Dataset.from_tensor_slices(dataset)
        data = data.batch(BATCH_SIZE)
        iterator = tf.data.Iterator.from_structure(data.output_types,data.output_shapes)
        img,label = iterator.get_next()
        return img,label,iterator

def make_data_initializer(data,iterator,BATCH_SIZE=64):
    with tf.name_scope('data'):
        data = tf.data.Dataset.from_tensor_slices(data)
        #data = data.shuffle(10000)
        data = data.batch(BATCH_SIZE)
        init = iterator.make_initializer(data)
        return init



def initialize_model(net,trainset,BATCH_SIZE=64):
    print('Initialization ... ')
    num_batches = len(trainset[0][0]) // BATCH_SIZE
    net.num_batches = num_batches
    X_holder,y_holder,iterator = make_iterator(trainset[0],BATCH_SIZE)
    
    
    net.set_fisher_graph(X_holder,y_holder)
    net.set_uncertain_prediction()
    out, log_probs, nll, kl_diver= net(X_holder, targets=y_holder, sample=True, n_samples=1, 
                              loss_function=lambda y, y_target: 
                                   tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_target, logits=y))
    net.set_vanilla_loss(log_probs,nll,num_batches)
    '''
    _, kl_log_probs, kl_nll, _= net(X_holder, targets=y_holder, sample=True, n_samples=1, 
                              loss_function=lambda y, y_target: 
                                   tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y))
    net.set_kl_loss(kl_log_probs,kl_nll,num_batches)
    '''
    #_, mode_kl_log_probs, mode_kl_nll, _= net(X_holder, targets=y_holder, sample=True, n_samples=1, drop_out=True,
    #                          loss_function=lambda y, y_target: 
    #                               tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_target, logits=y))
    #net.set_drop_loss(mode_kl_log_probs,mode_kl_nll,num_batches)
    
    net.summary()
    return iterator



def get_data_init(trainset,testsets,iterator):
    train_init = []
    test_init = []
    for t in range(len(trainset)):
        train_init.append(make_data_initializer(trainset[t],iterator))
    for t in range(len(testsets)):
        test_init.append(make_data_initializer(testsets[t],iterator))
    return train_init,test_init


def load_iterator(net,trainset,testsets):
    iterator = initialize_model(net,trainset)
    train_init,test_init = get_data_init(trainset,testsets,iterator)
    return train_init,test_init

