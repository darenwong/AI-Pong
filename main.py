# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 14:15:10 2018

@author: Daren
"""

import gym
import numpy as np
import tensorflow as tf
import time

learning_rate = 0.0001
n_hidden = 100


env = gym.make('Pong-ram-v0')


d = env.observation_space.shape[0]


def discount_and_normalize_rewards(episode_rewards, gamma):
    discounted_episode_rewards = np.zeros_like(episode_rewards, dtype=np.float32)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
    
    return discounted_episode_rewards


with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, (None, d))
    ep_actionx = tf.placeholder(tf.float32, (None, 2))
    ep_rewardx = tf.placeholder(tf.float32, None)

with tf.name_scope("fc1"):
    fc1 = tf.contrib.layers.fully_connected(inputs = x,
                                            num_outputs = n_hidden,
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer())

with tf.name_scope("fc2"):
    fc2 = tf.contrib.layers.fully_connected(inputs = fc1,
                                            num_outputs = n_hidden,
                                            activation_fn= tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer())

with tf.name_scope("fc3"):
    logit = tf.contrib.layers.fully_connected(inputs = fc2,
                                            num_outputs = 2,
                                            activation_fn= None,
                                            weights_initializer=tf.contrib.layers.xavier_initializer())

with tf.name_scope("softmax"):
    action_distribution = tf.nn.softmax(logit)

with tf.name_scope("loss"):
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=ep_actionx)
    loss = tf.reduce_mean(tf.multiply(ep_rewardx, entropy))

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


def main(mode):
    if mode == 'train':
        load_status = 0
        save_status = 1
        train = 1
        render = 0
    elif mode == 'play':
        load_status = 1
        save_status = 0
        train = 0
        render = 1

    saver = tf.train.Saver() 
    
    with tf.Session() as sess:
    
        sess.run(tf.global_variables_initializer())
        
        if load_status == 1:
            saver.restore(sess, '/tmp/pong_model9.ckpt')
            print("Model restored.")
    
        for i_episode in range(10000):
            
            observation = env.reset()
            ep_state = []
            ep_action = []
            ep_prob = []
            ep_reward = []
            tot_ep_reward = 0
            final_reward = []
            
            if i_episode%10 == 0 and save_status == 1:
                save_path = saver.save(sess, '/tmp/pong_model10.ckpt')
                print("Model saved in file: %s" % save_path)
            
            for t in range(100000):
              
                if render == 1:
                    env.render()
                    time.sleep(0.03)
                ep_state.append(list(observation/255))
    
    
                prob = sess.run(action_distribution ,feed_dict={x: [observation/255]})
    
                if np.random.rand() > prob[0][0]:
                    action = 2
                    ep_action.append([0,1])
    
                else:
                    action = 3
                    ep_action.append([1,0])
                
                ep_prob.append(list(prob[0]))
    
                observation, reward, done, info = env.step(action)
                tot_ep_reward += reward
                ep_reward.append(reward)
                
                gamma = 0.95
                if done:
                    break
    
            print("Episode ", i_episode, " finished after {} timesteps".format(t+1))
            final_reward = discount_and_normalize_rewards(ep_reward, gamma)
            final_reward = np.array(final_reward, dtype = np.float32)
            
            print('score: ', np.sum(ep_reward))
            print('loss: ', sess.run(loss, feed_dict={x: ep_state, ep_actionx: ep_action, ep_rewardx: final_reward}))
            
            if train == 1:
                sess.run(train_step, feed_dict={x: ep_state, ep_actionx: ep_action, ep_rewardx: final_reward})
                
#available mode are 'train' and 'play'
main('play')
