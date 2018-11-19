"""
The deep deterministic policy gradient model. Contains main training loop and deployment
"""
from __future__ import print_function

import os
import traceback
import json
import numpy as np
import tensorflow as tf

from collections import deque
from .replay_buffer import ReplayBuffer
from ..base_model import BaseModel


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax_Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


class DDPG(BaseModel):
    def __init__(self, env, sess, actor, critic, actor_noise, obs_normalizer=None, action_processor=None,
                 config_file='config/default.json',
                 model_save_path='weights/ddpg/ddpg.ckpt', summary_path='results/ddpg/'):
        with open(config_file) as f:
            self.config = json.load(f)
        assert self.config != None, "Can't load config file"
        np.random.seed(self.config['seed'])
        if env:
            env.seed(self.config['seed'])
        self.model_save_path = model_save_path
        self.summary_path = summary_path
        self.sess = sess
        # if env is None, then DDPG just predicts
        self.env = env
        self.actor = actor
        self.critic = critic
        self.action_take = np.zeros((3,))
        self.actor_noise = actor_noise
        self.obs_normalizer = obs_normalizer
        self.action_processor = action_processor
        self.summary_ops, self.summary_vars = build_summaries()

    def initialize(self, load_weights=True, verbose=True):
        """ Load training history from path. To be add feature to just load weights, not training states

        """
        if load_weights:
            try:
                variables = tf.global_variables()
                param_dict = {}
                saver = tf.train.Saver()
                saver.restore(self.sess, self.model_save_path)
                for var in variables:
                    var_name = var.name[:-2]
                    if verbose:
                        print('Loading {} from checkpoint. Name: {}'.format(var.name, var_name))
                    param_dict[var_name] = var
            except:
                traceback.print_exc()
                print('Build model from scratch')
                self.sess.run(tf.global_variables_initializer())
        else:
            print('Build model from scratch')
            self.sess.run(tf.global_variables_initializer())

    def train(self, save_every_episode=1, verbose=True, debug=False):
        """ Must already call intialize

        Args:
            save_every_episode:
            print_every_step:
            verbose:
            debug:

        Returns:

        """
        writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)

        self.actor.update_target_network()
        self.critic.update_target_network()

        np.random.seed(self.config['seed'])
        num_episode = self.config['episode']
        print(num_episode)
        batch_size = self.config['batch size']
        gamma = self.config['gamma']
        # Initialize replay buffer
        self.buffer = ReplayBuffer(self.config['buffer size'])
        np.set_printoptions(precision=2)

        # main training loop
        for i in range(num_episode):
            if verbose and debug:
                ("Episode: " + str(i) + " Replay Buffer " + str(self.buffer.count()))

            # Reset previous_observation
            previous_observation = self.env.reset()
            if self.obs_normalizer:
                previous_observation = self.obs_normalizer(previous_observation)

            ep_reward = 0
            ep_ave_max_q = 0

            for j in range(self.config['max step']):
                # print("step: {}, observation: {}".format(j, previous_observation))
                action = self.actor.predict(np.expand_dims(previous_observation, axis=0)).squeeze(
                    axis=0)

                if self.action_processor:
                    self.action_take = self.action_processor(action)
                else:
                    self.action_take = action
                if j % 10 == 0:
                    print(action)

                observation, reward, done, info = self.env.step(self.action_take)

                ep_reward += reward

                if self.obs_normalizer:
                    observation = self.obs_normalizer(observation)

                self.buffer.add(previous_observation, action, reward, done, observation)

                # Infer other states actions pairs
                for k in range(len(action) - 1):
                    action_p = np.zeros((len(action),))
                    action_p[k+1] = 1
                    _, reward_p, _, _ = self.env.step(action_p, simulation=1)
                    if reward_p > 0 or reward_p < -0.0003:
                        self.buffer.add(previous_observation, action_p, reward_p, done, observation)

                previous_observation = observation

                if done or j == self.config['max step'] - 1:
                    break

            if self.buffer.size() >= batch_size:
                # batch update
                s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(batch_size)
                # Calculate targets
                target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))

                y_i = []
                for k in range(batch_size):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = self.critic.train(
                    s_batch, a_batch, np.reshape(y_i, (batch_size, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = self.actor.predict(s_batch)
                grads = self.critic.action_gradients(s_batch, a_outs)
                self.actor.train(s_batch, grads[0])

                # Update target networks
                self.actor.update_target_network()
                self.critic.update_target_network()


            summary_str = self.sess.run(self.summary_ops, feed_dict={
                self.summary_vars[0]: ep_reward,
                self.summary_vars[1]: ep_ave_max_q / float(j)
            })

            writer.add_summary(summary_str, i)
            writer.flush()

            print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}'.format(i, np.exp(ep_reward), (ep_ave_max_q / float(j))))

            # save intermediate model
            if i % 100 == 0:
                intermediate_model_path = self.model_save_path + "_Episode_" + "{}".format(i)
                if not os.path.exists(intermediate_model_path):
                    os.makedirs(intermediate_model_path, exist_ok=True)

                saver = tf.train.Saver()
                saver.save(self.sess, intermediate_model_path + "/checkpoint.ckpt")

        self.save_model(verbose=True)

        print('Finish.')

    def predict(self, observation):
        """ predict the next action using actor model, only used in deploy.
            Can be used in multiple environments.

        Args:
            observation: (batch_size, num_stocks + 1, window_length)

        Returns: action array with shape (batch_size, num_stocks + 1)

        """
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)
        action = self.actor.predict(observation)
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def predict_single(self, observation):
        """ Predict the action of a single observation

        Args:
            observation: (num_stocks + 1, window_length)

        Returns: a single action array with shape (num_stocks + 1,)

        """
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)
        action = self.actor.predict(np.expand_dims(observation, axis=0)).squeeze(axis=0)
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def save_model(self, verbose=False):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

        saver = tf.train.Saver()
        model_path = saver.save(self.sess, self.model_save_path)
        print("Model saved in %s" % model_path)
