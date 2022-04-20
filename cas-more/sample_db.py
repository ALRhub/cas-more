from collections import deque
import numpy as np


class AbstractDatabase:
    @classmethod
    def get_default_config(cls, dim, buffer_fac=1.5):
        max_samples = int(np.maximum(np.ceil((buffer_fac * (1 + dim + int(dim * (dim + 1) / 2)))), (dim + 1) * 8))

        buffer_config = {"buffer_fac": buffer_fac,
                         "max_samples": max_samples,
                         }
        return buffer_config

    def __init__(self, size):
        self.size = size
        self.data_x = deque(maxlen=size)
        self.data_y = deque(maxlen=size)
        self.current_samples = None
        self.current_rewards = None

    @property
    def buffer_size(self):
        return len(self.data_y)


class SimpleSampleDatabase(AbstractDatabase):
    def add_data(self, data_x, data_y):
        self.current_samples = data_x
        self.current_rewards = data_y
        self.data_x.extend(data_x)
        self.data_y.extend(data_y)

    def get_data(self):
        data_x = np.vstack(self.data_x)
        data_y = np.vstack(self.data_y)
        return data_x, data_y
