from torch.utils.data.sampler import Sampler
import numpy as np


class ChunkedRandomSampler(Sampler):
    """Splits a dataset into smaller chunks (mainly to re-define what is considered an 'epoch').
       Samples elements randomly from a given list of indices, without replacement.
       If a chunk would be underpopulated, it's filled up with rest-samples.

    Arguments:
        data_source (Dataset): a dataset
        chunk_size      (int): how large a chunk should be
    """

    def __init__(self, data_source, chunk_size):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.i = 0
        self.N = len(self.data_source)
        # re-did this as numpy permutation, b/c FramedSignals do not like
        # torch tensors as indices ...
        # self.perm = torch.randperm(self.N)
        self.perm = np.random.permutation(self.N)

    def __iter__(self):
        rest = len(self.perm) - (self.i + self.chunk_size)
        if rest == 0:
            self.i = 0
            self.perm = np.random.permutation(self.N)
        elif rest < 0:
            # works b/c rest is negative
            carryover = self.chunk_size + rest
            self.i = 0
            self.perm = np.hstack([self.perm[-carryover:], np.random.permutation(self.N)])

        chunk = self.perm[self.i: self.i + self.chunk_size]
        self.i += self.chunk_size
        return iter(chunk)

    def __len__(self):
        return self.chunk_size
