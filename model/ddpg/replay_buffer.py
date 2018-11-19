"""
Source: https://github.com/vermouth1992/deep-learning-playground/blob/master/tensorflow/ddpg/replay_buffer.py
"""
from collections import deque
import random
import os
import csv
import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def store(self):
        data_folder = "/buffer_data"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        # Write screen features
        data_path = data_folder + '/' + 'experience_replay' + '.txt'
        mode = 'a+' if os.path.exists(data_path) else 'w+'
        with open(data_path, mode, newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.buffer)

    def restore(self, data_path):
        with open(data_path, 'r') as f:
            self.buffer = [line[:-1].strip('""') for line in f]
        return self.buffer