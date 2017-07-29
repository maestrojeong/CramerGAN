'''
CRAMER_GAN(WGAN-gp)

Updated on 2017.07.29
Author : Yeonwoo Jeong
'''
from ops import mnist_for_gan, optimizer, clip, get_shape, softmax_cross_entropy, sigmoid_cross_entropy
from config import CramerGANConfig, SAVE_DIR, PICTURE_DIR
from utils import show_gray_image_3d, make_gif, create_dir
from nets import GenConv, DisConv, Critic
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging
import glob
import os


logging.basicConfig(format = "[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def sample_z(z_size, z_dim):
    return np.random.uniform(low=-1, high=1, size= [z_size, z_dim])

class InfoGAN(CramerGANConfig):
    def __init__(self):
        CramerGANConfig.__init__(self)
        logger.info("Building model starts...")
        tf.reset_default_graph()
        self.generator = GenConv(name ='g_conv', batch_size=self.batch_size)
        self.discriminator = DisConv(name='d_conv')
        self.critic = Critic(self.discriminator)
        self.dataset = mnist_for_gan()
        
        self.X = tf.placeholder(tf.float32, shape = [self.batch_size, self.x_size, self.x_size, self.x_channel])
        self.Z1 = tf.placeholder(tf.float32, shape = [self.batch_size, self.z_dim])
        self.Z2 = tf.placeholder(tf.float32, shape = [self.batch_size, self.z_dim])

        dummy = self.discriminator(self.X)
        
        self.G_sample1 = self.generator(tf.concat([self.Z1], axis=1))
        self.G_sample2 = self.generator(tf.concat([self.Z2], axis=1), reuse = True)

        # x_hat = epsilon*x_real + (1-epsilon)*x_gen
        self.epsilon = tf.random_uniform(shape=[self.batch_size, 1, 1, 1],minval=0, maxval=1)# epsilon : sample from uniform [0,1]
        self.linear_ip = self.epsilon*self.X + (1-self.epsilon)*self.G_sample1
        
        # Gradient penalty
        self.D_ip = self.critic(self.linear_ip, self.G_sample2)
        self.gradient = tf.gradients(self.D_ip, [self.linear_ip])[0]
        self.gradient_penalty = tf.reduce_mean(tf.square(tf.norm(self.gradient, axis=1) - 1.))

        # error from classfication + error from regression
        self.G_loss = tf.reduce_mean(self.critic(self.X, self.G_sample2)-self.critic(self.G_sample1, self.G_sample2))        
        self.D_loss = -self.G_loss +self.lamb*self.gradient_penalty

        self.generator.print_vars()
        self.discriminator.print_vars()

        self.D_optimizer = optimizer(self.D_loss, self.discriminator.vars)
        self.G_optimizer = optimizer(self.G_loss, self.generator.vars)

        logger.info("Building model done.")
        self.sess = tf.Session()
        
    def initialize(self):
        """Initialize all variables in graph"""
        logger.info("Initializing model parameters")
        self.sess.run(tf.global_variables_initializer())

    def restore(self):
        """Restore all variables in graph"""
        logger.info("Restoring model starts...")
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(SAVE_DIR))
        logger.info("Restoring model done.")     
    
    def sample_data(self):
        """sampling for data
        Return:
            X_sample, z1_sample, z2_sample 
        """
        X_sample = self.dataset(self.batch_size)
        z_sample1 = sample_z(self.batch_size, self.z_dim)
        z_sample2 = sample_z(self.batch_size, self.z_dim)
        return X_sample, z_sample1, z_sample2

    def train(self, train_epochs):
        count = 0
        for epoch in tqdm(range(train_epochs), ascii = True, desc = "batch"):
            if epoch < 25:
                d_iter = 100
            else:
                d_iter = 5
            for _ in range(d_iter):
                X_sample, z_sample1, z_sample2 = self.sample_data()
                self.sess.run(self.D_optimizer, feed_dict = {self.X : X_sample, self.Z1 : z_sample1, self.Z2 : z_sample2})
            
            for _ in range(1):
                X_sample, z_sample1, z_sample2 = self.sample_data()
                self.sess.run(self.G_optimizer, feed_dict = {self.X : X_sample, self.Z1 : z_sample1, self.Z2 : z_sample2})
            
            for _ in range(0):
                X_sample, z_sample1, z_sample2 = self.sample_data()
                self.sess.run(self.Q_optimizer, feed_dict = {self.Z1 : z_sample1, self.Z2 : z_sample2})

            if epoch % self.log_every == self.log_every-1:
                X_sample, z_sample1, z_sample2 = self.sample_data()
                D_loss = self.sess.run(self.D_loss, feed_dict = {self.X : X_sample, self.Z1 : z_sample1, self.Z2 : z_sample2})
                G_loss = self.sess.run(self.G_loss, feed_dict = {self.X : X_sample, self.Z1 : z_sample1, self.Z2 : z_sample2})
                
                count+=1
                for index in range(2):
                    X_sample, z_sample1, z_sample2 = self.sample_data()
                    gray_3d = self.sess.run(self.G_sample1, feed_dict = {self.Z1 : z_sample1}) # self.batch_size x 28 x 28 x 1
                    gray_3d = np.squeeze(gray_3d)#self.batch_size x 28 x 28
                	# Store generated image on PICTURE_DIR
                    fig = show_gray_image_3d(gray_3d, col=10, figsize = (50, 50), dataformat = 'CHW')
                    fig.savefig(PICTURE_DIR+"%s_%d.png"%(str(count).zfill(3), index))
                    plt.close(fig)

                logger.info("Epoch({}/{}) D_loss : {}, G_loss : {}".format(epoch+1, train_epochs, D_loss, G_loss))
                # Save model
                saver=tf.train.Saver(max_to_keep = 10)
                saver.save(self.sess, os.path.join(SAVE_DIR, 'model'), global_step = epoch+1)
                logger.info("Model save in %s"%SAVE_DIR)


if __name__=='__main__':
    create_dir(SAVE_DIR)
    create_dir(PICTURE_DIR)
    infogan = InfoGAN()
    infogan.initialize()
    infogan.train(100000)

    for index in range(2):
        images_path = glob.glob(os.path.join(PICTURE_DIR, '*_%d.png'%index))
        gif_path = os.path.join(PICTURE_DIR, '%d.gif'%index)
        make_gif(sorted(images_path), gif_path)
