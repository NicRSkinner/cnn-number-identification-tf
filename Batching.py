import numpy as np


class Batching:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.last_batch_start = 0
        self.last_batch_end = 0

    def next_batch(self, batch_size):
        self.last_batch_start = self.last_batch_end
        self.last_batch_end = self.last_batch_start + batch_size

        yield self.x[self.last_batch_start % self.x.size:self.last_batch_end % self.x.size]
        yield self.y[self.last_batch_start % self.y.size:self.last_batch_end % self.y.size]
