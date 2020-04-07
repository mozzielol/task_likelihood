import tensorflow as tf
import sonnet as snt
import numpy as np


class BNNLayer(snt.AbstractModule):

    def __init__(self, n_inputs, n_outputs, init_mu=0.0, init_rho=0.0, num_task=2, name="BNNLayer"):
        super(BNNLayer, self).__init__(name=name)
        self.num_task = num_task
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.w_mean = tf.Variable(init_mu*tf.ones([self.n_inputs, self.n_outputs]))
        self.w_rho = tf.Variable(init_rho*tf.ones([self.n_inputs, self.n_outputs]))
        self.w_sigma = tf.log(1.0 + tf.exp(self.w_rho))
        self.w_distr = tf.distributions.Normal(loc=self.w_mean, scale=self.w_sigma)

        self.b_mean = tf.Variable(init_mu*tf.ones([self.n_outputs]))
        self.b_rho = tf.Variable(init_rho*tf.ones([self.n_outputs]))
        self.b_sigma = tf.log(1.0 + tf.exp(self.b_rho))
        self.b_distr = tf.distributions.Normal(loc=self.b_mean, scale=self.b_sigma)

    

        self.w_prior_mean = tf.Variable(tf.zeros_like(self.w_mean,dtype=tf.float32),trainable=False)
        self.w_prior_sigma = tf.Variable(tf.ones_like(self.w_sigma,dtype=tf.float32),trainable=False)
        self.b_prior_mean = tf.Variable(tf.zeros_like(self.b_mean,dtype=tf.float32),trainable=False)
        self.b_prior_sigma = tf.Variable(tf.ones_like(self.b_sigma,dtype=tf.float32),trainable=False)
        self.w_prior_distr = tf.distributions.Normal(loc=self.w_prior_mean, scale=self.w_prior_sigma)
        self.b_prior_distr = tf.distributions.Normal(loc=self.b_prior_mean, scale=self.b_prior_sigma)

    def get_mean_list(self):
        return [self.w_mean,self.b_mean]

    def get_var_list(self):
        return [self.w_rho,self.b_rho]

    def get_prior_mean_list(self):
        return [self.w_prior_mean,self.b_prior_mean]

    def get_prior_var_list(self):
        return [self.w_prior_sigma,self.b_prior_sigma]




    def get_learned_dist(self):
        return self.learned_mean, self.learned_var

    def get_sampled_weights(self):
        #w = self.w_mean + (self.w_sigma * tf.random_normal([self.n_inputs, self.n_outputs], 0.0, 1.0, tf.float32))
        #b = self.b_mean + (self.b_sigma * tf.random_normal([self.n_outputs], 0.0, 1.0, tf.float32))
        w = self.w_distr.sample()
        b = self.b_distr.sample()
        return w,b


    def _build(self, inputs, activation,sample=False, drop_out=False,pre_w=None,pre_b=None):
        """
        Constructs the graph for the layer.

        Args:
          inputs: `tf.Tensor` input to be used by the MLP
          sample: boolean; whether to compute the output of the MLP by sampling its weights from their posterior or by returning a MAP estimate using their mean value

        Returns:
          output: `tf.Tensor` output averaged across n_samples, if sample=True, else MAP output
          log_probs: `tf.Tensor` KL loss of the network
        """

        if sample:
            w = self.w_distr.sample()
            b = self.b_distr.sample()
        else:
            w = self.w_mean
            b = self.b_mean

        z = tf.matmul(inputs,w) + b
        z = activation(z)

            

        log_probs_w = self.w_distr.log_prob(w) - self.w_prior_distr.log_prob(w)
        log_probs_b = self.b_distr.log_prob(b) - self.b_prior_distr.log_prob(b)
        #2 - prior
        #1 - posterior
        #        kl_divergence = tf.log(sigma2) - tf.log(sigma1) + ((tf.square(sigma1) +
                                                            #tf.square(mean1 - mean2)) / (2 * tf.square(sigma2))) - 0.5

        return z, log_probs_w,log_probs_b


