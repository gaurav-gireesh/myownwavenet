import mxnet as mx
from mxnet.gluon.data import Dataset
from mxnet import ndarray
from scipy.io import wavfile
from .ggtransforms import encode_mu_law, decode_mu_law
import os

class WavenetDataset(Dataset):
    """A dataset that combines multiple dataset-like objects, e.g.
    Datasets, lists, arrays, etc.

    The i-th sample is defined as `(x1[i], x2[i], ...)`.

    Parameters
    ----------
    *args : one or more dataset-like objects
        The data arrays.
    """
    def __init__(self, file_nm, seq_size, mu, ctx, *args):
        self._file_nm = file_nm
        _, data = self._load_wav()
        self._seq_size = seq_size
        self._mu = mu
        self._ctx = ctx
        self._length = data.shape[0]-seq_size

        div = max(data.max(),abs(data.min()))
        data = data/div
        self._data = data

        #dataload here...


        # assert len(args) > 0, "Needs at least 1 arrays"
        # self._length = len(args[0])
        # self._data = []
        # for i, data in enumerate(args):
        #     assert len(data) == self._length, \
        #         "All arrays must have the same length; array[0] has length %d " \
        #         "while array[%d] has %d." % (self._length, i+1, len(data))
        #     if isinstance(data, ndarray.NDArray) and len(data.shape) == 1:
        #         data = data.asnumpy()
        #     self._data.append(data)


    #Loading using scipy's wavfile library
    def _load_wav(self):
        fs, data = wavfile.read(os.getcwd()+'/data/'+self._file_nm)
        return  fs, data


    def __getitem__(self, idx):
        # print("Called getitem")
        start = idx
        ys = self._data[start:start+self._seq_size]
        ys = encode_mu_law(ys,self._mu)
        return mx.nd.array(ys[:self._seq_size],ctx=self._ctx)
        # if len(self._data) == 1:
        #     return self._data[0][idx]
        # else:
        #     return tuple(data[idx] for data in self._data)

    def __len__(self):
        return self._length