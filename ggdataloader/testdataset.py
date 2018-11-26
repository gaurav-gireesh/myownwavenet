from ggDataLoader import DataLoader
from ggDataset import WavenetDataset
import mxnet as mx

#  def __init__(self, file_nm, seq_size, mu, ctx, *args):
       
wav_filename = '/Users/gireeg/Desktop/Internship/repos/WaveNet-gluon/data/parametric-2.wav'

mu = 128
seq_size = 20000

wds = WavenetDataset(   wav_filename, seq_size=seq_size, mu=mu, ctx = mx.cpu())
print(len(wds), " returned by the __length__() = ",wds.__len__())
print("The actual number of samples in the wav file then: ", len(wds._data))

# print(wds[0])
# print(wds.__getitem__(62081))
# print (wds._data[len(wds._data)-2])

dl2 = DataLoader(wds, batch_size=5, shuffle=True)

for data in dl2:
    print(data.shape)
    break


