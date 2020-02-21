from h5py import File
from matplotlib.pyplot import *


fh = File('profiles/profiles_s1.h5','r')

z       = fh['scales/z/1'][:]

theta   = fh['tasks']['tz'][-1,0,0,:]
w_rms   = fh['tasks']['w_rms'][-1,0,0,:]
u_rms   = fh['tasks']['u_rms'][-1,0,0,:]

fig, ax = subplots()

plot(z,w_rms/u_rms)
show()

fig, ax = subplots()

plot(z,theta)
show()
