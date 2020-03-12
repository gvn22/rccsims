from h5py import File
from matplotlib.pyplot import *
from numpy import pi,average,linspace

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
fh = File('ra20pr1/series/series_s1.h5','r')

t_pr1       = fh['scales/sim_time'][:]
nu_pr1      = fh['tasks/th_one'][:,0,0,0]
tz_pr1      = fh['tasks/th_half'][2:,0,0,0]

fh.close()

fh = File('ra20pr8/series/series_s1.h5','r')

t_pr8       = fh['scales/sim_time'][:]
nu_pr8      = fh['tasks/th_one'][:,0,0,0]
tz_pr8      = fh['tasks/th_half'][2:,0,0,0]

fh.close()

fh = File('ra80pr1/series/series_s1.h5','r')

t_ra80pr1       = fh['scales/sim_time'][:]
nu_ra80pr1      = fh['tasks/th_one'][:,0,0,0]
tz_ra80pr1      = fh['tasks/th_half'][2:,0,0,0]

fh.close()

fig, ax = subplots(figsize=(2*pi,pi))

plot(t_pr1,nu_pr1,'k',label=r'${Pr} = 1$')
plot(t_pr8,nu_pr8,'k-.',label=r'${Pr} = \infty$')
legend(loc=4,framealpha=0)

xlabel(r'$t$',fontsize=11)
ylabel(r'${Nu} = -\partial_Z \overline{\theta} |_{Z = 1}$',fontsize=11)
xlim(0,20)

savefig('figures/nu.png',bbox_inches='tight',dpi=120)

fig, ax = subplots(figsize=(pi,pi))

ra = [8.6956,10,20,40,80,160]
nu = [1.0,1.3253,5.3583,19.177,59.291,164.06]
tz = [8.6956,1.3253,0.31080,0.14933,0.07403,0.03693]

tz_pr = [average(tz_pr1),average(tz_ra80pr1)]
plot([20,80],tz_pr,'ko--',mfc='none',label=r'${Pr} = 1$')
plot(20,average(tz_pr1),'ks-.',mfc='none',label=r'${Pr} = \infty$')

plot(ra,tz,'k-',label='Sprague single mode')


print (tz_ra80pr1)

legend(loc=3,framealpha=0)
xlabel(r'$\widetilde{Ra}$',fontsize=11)
ylabel(r'${\langle - \partial_Z \overline{\theta} |_{Z = 1/2} \rangle_t}$',fontsize=11)

yscale('log')
xscale('log')
xlim(10,200)
ylim(0.02,1.2)
savefig('figures/tz.png',bbox_inches='tight',dpi=120)
