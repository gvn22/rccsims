from h5py import File
from matplotlib.pyplot import *


fh = File('series/series_s1.h5','r')

t       = fh['scales/sim_time'][:]
nu      = fh['tasks/Nu'][:,0,0,0]

fig, ax = subplots()

plot(t,nu)
show()