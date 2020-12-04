from h5py import File
from matplotlib.pyplot import *
from numpy import pi, average

fh = File('snapshots/snapshots_s1.h5','r')

ze = fh['/tasks/ze bot'][0,:,:,0]
t = fh['/scales/sim_time']

fig,ax = subplots()

imshow(ze,origin='bottom',cmap='Greys')

xticks([])
yticks([])
print (t[:])
savefig('ze.png',dpi=256,bbox_inches='tight')