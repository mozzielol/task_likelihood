import tensorflow as tf
import sonnet as snt
from tqdm import tqdm
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from bnn.BNNLayer import * 

class BNN_MLP(snt.AbstractModule):
    """
    Implementation of an Bayesian MLP with structure [n_inputs] -> hidden_units -> [n_outputs], with a diagonal gaussian
    posterior. The weights are initialized to mean init_mu mean and standard deviation log(1+exp(init_rho)).
    """

    def __init__(self, n_inputs, n_outputs, hidden_units=[],num_task=2, init_mu=0.0, init_rho=-3.0, activation=tf.nn.relu,
                         last_activation=tf.identity,name="BNN_MLP"):
        super(BNN_MLP, self).__init__(name=name)
        
        self.num_runs = 1
        self.x_placeholder = tf.placeholder(tf.float32, shape=[None, n_inputs])
        self.hidden_units = [n_inputs] + hidden_units + [n_outputs]

        self.num_task = num_task
        self.activation = activation
        self.last_activation = last_activation
        
        self.gstep = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
        self.lams = tf.get_variable('lams',initializer=tf.constant(0.1),trainable=False)
        #Initialize the storage
        self.tensor_mean = []
        self.tensor_var = []

        self.params_mean = {}
        self.params_var = {}


        self.layers = []
        
        self.y_placeholder = tf.placeholder(tf.float32, shape=[None, n_outputs])


        for i in range(1, len(self.hidden_units)):
            self.layers.append( BNNLayer(self.hidden_units[i-1], self.hidden_units[i], init_mu=init_mu, init_rho=init_rho,num_task=num_task))
            self.tensor_mean += self.layers[i-1].get_mean_list()
            self.tensor_var += self.layers[i-1].get_var_list()

        


    def initialize_default_params(self,sess):
        sess.run( tf.global_variables_initializer() )
        self.params_mean = {}
        self.params_var = {}
        
        

    def reset(self,sess):
        sess.run( tf.global_variables_initializer() )


    def transform_var(self,var):
        return np.log(1.0 + np.exp(var))

    def retransform_var(self,var):
        return np.log(np.exp(var) - 1.0)



    def store_params(self,num):
        #if num == 1 : pass
        mean_list = []
        var_list = []
        for v in self.tensor_mean:
            mean_list.append(v.eval())
        for v in self.tensor_var:
            var_list.append(v.eval())
        self.params_mean[num] = mean_list
        self.params_var[num] = var_list





    def set_task_params(self,sess,num):
        for idx in range(len(self.params_mean[num])):
            sess.run(self.tensor_mean[idx].assign(self.params_mean[num][idx]))
            sess.run(self.tensor_var[idx].assign(self.params_var[num][idx]))



    def set_fisher_graph(self,x,y_):
        inputs = x
        out_test_deterministic, _, _,_ = self(inputs, sample=False, loss_function=None)
        prediction = tf.cast(tf.argmax(out_test_deterministic, 1), tf.int64)
        equality = tf.equal(prediction, tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
                
        


    def set_uncertain_prediction(self):
        marks = 0.0
        x = self.x_placeholder
        y_ = self.y_placeholder

        inputs = x
        for i in range(len(self.layers)):
            if i == len(self.layers)-1:
                inputs, _ ,_= self.layers[i](inputs,self.last_activation,sample=True,drop_out=False)
            else:
                inputs, _ ,_= self.layers[i](inputs,self.activation,sample=True,drop_out=False)
        self.predictions = tf.nn.softmax(inputs)
        
        correct_prediction = tf.equal(tf.argmax(self.predictions,1), tf.argmax(y_,1))
        self.em_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





    def _build(self, inputs,sample=False, n_samples=1, targets=None, drop_out=False,
        loss_function=lambda y, y_target: tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_target, logits=y)):
        """
        Constructs the MLP graph.

        Args:
          inputs: `tf.Tensor` input to be used by the MLP
          sample: boolean; whether to compute the output of the MLP by sampling its weights from their posterior or by returning a MAP estimate using their mean value
          n_samples: number of sampled networks to average output of the MLP over
          targets: target outputs of the MLP, used to compute the loss function on each sampled network
          loss_function: lambda function to compute the loss of the network given its output and targets.

        Returns:
          output: `tf.Tensor` output averaged across n_samples, if sample=True, else MAP output
          log_probs: `tf.Tensor` KL loss of the network
          avg_loss: `tf.Tensor` average loss across n_samples, computed using `loss_function'
        """

        log_probs = 0.0
        avg_loss = 0.0
        kl_diver = []
        pre_w = None
        pre_b = None
        if not sample:
            n_samples = 1

        output = 0.0 ## avg. output logits
        for ns in range(n_samples):

            x = inputs
            for i in range(len(self.layers)):
                    
                if i == len(self.layers)-1:
                    x, log_probs_w,log_probs_b = self.layers[i](x,self.last_activation,sample,drop_out=False)
                else:
                    x, log_probs_w,log_probs_b = self.layers[i](x,self.activation,sample,drop_out=False)
                kl_diver.append(log_probs_w)
                kl_diver.append(log_probs_b)
                log_probs += tf.reduce_sum(log_probs_w) + tf.reduce_sum(log_probs_b)

            output += x

            if targets is not None:
                if loss_function is not None:
                    loss = tf.reduce_mean(loss_function(x, targets), 0)

                    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=x))
                    #loss = 0.5*tf.reduce_mean(tf.reduce_sum( tf.square(targets-x), 1), 0)
                    avg_loss += loss


        log_probs /= n_samples
        avg_loss /= n_samples
        output /= n_samples


        return output, log_probs, avg_loss,kl_diver


    def set_vanilla_loss(self,log_probs,nll,num_batches):
        self.loss = (self.lams/2)*log_probs/num_batches + nll #+ (self.lams*50/2) * sigma_loss #+ uncertain_loss
        with tf.variable_scope("wrapped_optimizer"):
            optim = tf.train.AdamOptimizer(learning_rate=0.001)
            self.train_op = optim.minimize( self.loss ,global_step=self.gstep)
        wrapped_opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "wrapped_optimizer")
        self.init_opt_vars = tf.variables_initializer(wrapped_opt_vars)



    def summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss',self.loss)
            tf.summary.scalar('accuracy',self.accuracy)
            tf.summary.histogram('histogram',self.loss)
            self.summary_op = tf.summary.merge_all()



    def _st_smooth(self,var_idx,x_v,y_v=None,n_component=1,thresh_hold=0.3,dp=False,alpha=0.5):
        mixture_dist = []

        dist = [] # second
        for task_idx in range(self.num_task):
            if y_v is not None:
                mean = self.params_mean[task_idx][var_idx][x_v][y_v]
                var = self.transform_var(self.params_var[task_idx][var_idx][x_v][y_v])
                
            else:
                mean = self.params_mean[task_idx][var_idx][x_v]
                var = self.transform_var(self.params_var[task_idx][var_idx][x_v])
        
            nor = np.random.normal(mean,var,200) #second
            dist.append(nor)    #second
        dist = np.array(np.asmatrix(np.concatenate(dist)).T)    #second
        
        

        gmm = GMM( max_iter=500,  n_components=n_component, covariance_type='spherical')
        #gmm.fit(sample)
        gmm.fit(dist)
        
        new_idx_list = []
        for task_idx in range(self.num_task):
            if y_v is not None:
                predict_probability = gmm.predict_proba(np.array(self.params_mean[task_idx][var_idx][x_v][y_v]).reshape(-1,1))
            else:
                predict_probability = gmm.predict_proba(np.array(self.params_mean[task_idx][var_idx][x_v]).reshape(-1,1))
            f_ = True
            while f_:
                if gmm.weights_[np.argmax(predict_probability)] > thresh_hold:
                    new_idx = np.argmax(predict_probability)
                    f_ = False
                else:
                    predict_probability[0][np.argmax(predict_probability)] = 0.0
                    
            if new_idx in new_idx_list:
                self.num_merged_params += 1
            new_idx_list.append(new_idx)
            if y_v is not None:
                self.params_mean[task_idx][var_idx][x_v][y_v] = gmm.means_[new_idx]
                self.params_var[task_idx][var_idx][x_v][y_v] = self.retransform_var(gmm.covariances_[new_idx])
            else:
                self.params_mean[task_idx][var_idx][x_v] = gmm.means_[new_idx]
                self.params_var[task_idx][var_idx][x_v] = self.retransform_var(gmm.covariances_[new_idx])


    def st_smooth(self,n_component=1,dp=True,thresh_hold=0.3,alpha=0.5):
        self.num_merged_params = 0
        #The range of len(params)
        _step = 0
        for var_idx in tqdm(range(len(self.params_mean[0]))):
            for x_v in range(len(self.params_mean[0][var_idx])):
                print('Step %d'%_step,end='\r')
                _step += 1
                try:
                    for y_v in range(len(self.params_mean[0][var_idx][x_v])):
                        self._st_smooth(var_idx,x_v,y_v=y_v,n_component=n_component,thresh_hold=thresh_hold,dp=dp,alpha=alpha)

                except TypeError:
                    self._st_smooth(var_idx,x_v,n_component=n_component,thresh_hold=thresh_hold,dp=dp,alpha=alpha)
                    


