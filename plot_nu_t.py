from h5py import File
from matplotlib.pyplot import *
from numpy import pi,average,linspace

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
fh = File('series/series_s1.h5','r')

t  = fh['scales/sim_time'][:]
Nu = fh['tasks/Nu'][:,0,0,0]

fh.close()

fig, ax = subplots(figsize=(2*pi,pi))

plot(t,Nu,'k')
xlabel(r'$t$',fontsize=11)
ylabel(r'${Nu}$',fontsize=11)

show()
# savefig('figures/nu_t.png',bbox_inches='tight',dpi=120)
