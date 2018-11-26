import pickle
import io
import sys
import multiprocessing
import multiprocessing.queues
from multiprocessing.reduction import ForkingPickler
import threading
import numpy as np

try:
    import multiprocessing.resource_sharer
except ImportError:
    pass

import mxnet as mx
from mxnet import nd, context, gluon
from mxnet.gluon.data import sampler as _sampler

# from . import sampler as _sampler
# from ... import nd, context
# from ...recordio import MXRecordIO



def default_batchify_fn(data):
    """Collate data into batch."""
    if isinstance(data[0], nd.NDArray):
        return nd.stack(*data)
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [default_batchify_fn(i) for i in data]
    else:
        data = np.asarray(data)
        return nd.array(data, dtype=data.dtype)

def _as_in_context(data, ctx):
    """Move data into new context."""
    if isinstance(data, nd.NDArray):
        return data.as_in_context(ctx)
    elif isinstance(data, (list, tuple)):
        return [_as_in_context(d, ctx) for d in data]
    return data

class DataLoader(object):
    """Loads data from a dataset and returns mini-batches of data.

    Parameters
    ----------
    dataset : Dataset
        Source dataset. Note that numpy and mxnet arrays can be directly used
        as a Dataset.
    batch_size : int
        Size of mini-batch.
    shuffle : bool
        Whether to shuffle the samples.
    sampler : Sampler
        The sampler to use. Either specify sampler or shuffle, not both.
    last_batch : {'keep', 'discard', 'rollover'}
        How to handle the last batch if batch_size does not evenly divide
        `len(dataset)`.

        keep - A batch with less samples than previous batches is returned.
        discard - The last batch is discarded if its incomplete.
        rollover - The remaining samples are rolled over to the next epoch.
    batch_sampler : Sampler
        A sampler that returns mini-batches. Do not specify batch_size,
        shuffle, sampler, and last_batch if batch_sampler is specified.
    batchify_fn : callable
        Callback function to allow users to specify how to merge samples
        into a batch. Defaults to `default_batchify_fn`::

            def default_batchify_fn(data):
                if isinstance(data[0], nd.NDArray):
                    return nd.stack(*data)
                elif isinstance(data[0], tuple):
                    data = zip(*data)
                    return [default_batchify_fn(i) for i in data]
                else:
                    data = np.asarray(data)
                    return nd.array(data, dtype=data.dtype)

    num_workers : int, default 0
        The number of multiprocessing workers to use for data preprocessing.
    pin_memory : boolean, default False
        If ``True``, the dataloader will copy NDArrays into pinned memory
        before returning them. Copying from CPU pinned memory to GPU is faster
        than from normal CPU memory.
    """
    def __init__(self, dataset, batch_size=None, shuffle=False, sampler=None,
                 last_batch=None, batch_sampler=None, batchify_fn=None,
                 num_workers=0, pin_memory=False):
        self._dataset = dataset
        self._pin_memory = pin_memory

        if batch_sampler is None:
            if batch_size is None:
                raise ValueError("batch_size must be specified unless " \
                                 "batch_sampler is specified")
            if sampler is None:
                if shuffle:
                    sampler = _sampler.RandomSampler(len(dataset))
                else:
                    sampler = _sampler.SequentialSampler(len(dataset))
            elif shuffle:
                raise ValueError("shuffle must not be specified if sampler is specified")

            batch_sampler = _sampler.BatchSampler(
                sampler, batch_size, last_batch if last_batch else 'keep')
        elif batch_size is not None or shuffle or sampler is not None or \
                last_batch is not None:
            raise ValueError("batch_size, shuffle, sampler and last_batch must " \
                             "not be specified if batch_sampler is specified.")

        self._batch_sampler = batch_sampler
        self._num_workers = num_workers if num_workers >= 0 else 0
        if batchify_fn is None:
            # if num_workers > 0:
            #     self._batchify_fn = default_mp_batchify_fn
            # else:
            self._batchify_fn = default_batchify_fn
        else:
            self._batchify_fn = batchify_fn

    def __iter__(self):
        if self._num_workers == 0:
            def same_process_iter():
                for batch in self._batch_sampler:
                    ret = self._batchify_fn([self._dataset[idx] for idx in batch])
                    if self._pin_memory:
                        ret = _as_in_context(ret, context.cpu_pinned())
                    yield ret
            return same_process_iter()

        # multi-worker
        # return _MultiWorkerIter(self._num_workers, self._dataset,
        #                         self._batchify_fn, self._batch_sampler, self._pin_memory)

    def __len__(self):
        return len(self._batch_sampler)