from h5py import File
from matplotlib.pyplot import *
from numpy import pi, average
from scipy.integrate import cumtrapz

fh = File('ra20pr1/profiles/profiles_s1.h5','r')

z       = fh['scales/z/1'][:]

theta_ra20pr1   = average(fh['tasks']['tz'][20:,0,0,:],axis = 0)
w_rms_ra20pr1   = fh['tasks']['w_rms'][-1,0,0,:]
t_rms_ra20pr1   = fh['tasks']['tf_rms'][-1,0,0,:]
u_rms_ra20pr1   = fh['tasks']['u_rms'][-1,0,0,:]
v_rms_ra20pr1   = fh['tasks']['v_rms'][-1,0,0,:]
ze_rms_ra20pr1   = fh['tasks']['ze_rms'][-1,0,0,:]

fh.close()

fh = File('ra20pr8/profiles/profiles_s1.h5','r')

z       = fh['scales/z/1'][:]

theta_ra20pr8   = average(fh['tasks']['tz'][20:,0,0,:],axis = 0)
w_rms_ra20pr8   = fh['tasks']['w_rms'][-1,0,0,:]
t_rms_ra20pr8   = fh['tasks']['tf_rms'][-1,0,0,:]
u_rms_ra20pr8   = fh['tasks']['u_rms'][-1,0,0,:]
v_rms_ra20pr8   = fh['tasks']['v_rms'][-1,0,0,:]
ze_rms_ra20pr8   = fh['tasks']['ze_rms'][-1,0,0,:]

fh.close()

fh = File('ra80pr1/profiles/profiles_s1.h5','r')

z_80       = fh['scales/z/1'][:]

theta_ra80pr1   = average(fh['tasks']['tz'][20:,0,0,:],axis = 0)
w_rms_ra80pr1   = fh['tasks']['w_rms'][-1,0,0,:]
t_rms_ra80pr1   = fh['tasks']['tf_rms'][-1,0,0,:]
u_rms_ra80pr1   = fh['tasks']['u_rms'][-1,0,0,:]
v_rms_ra80pr1   = fh['tasks']['v_rms'][-1,0,0,:]
ze_rms_ra80pr1   = fh['tasks']['ze_rms'][-1,0,0,:]

fh.close()


fig, ax = subplots(figsize=(pi,pi))

xlim(0,12)
ylim(0.5,1.0)

plot(ze_rms_ra20pr1[31:],z[31:],'k-',label=r'${Pr} = 1$')
plot(ze_rms_ra20pr8[31:],z[31:],'k-.',label=r'${Pr} = \infty$')

xlabel(r'$[\omega_3^\prime]_{RMS}$',fontsize=11)
ylabel(r'$Z$',fontsize=11)
legend(loc=4,framealpha=0)

savefig('ze_rms.png',dpi=100,bbox_inches='tight')

fig, ax = subplots(figsize=(pi,pi))

xlim(0,5)
ylim(0.5,1.0)

plot(w_rms_ra20pr1[31:],z[31:],'k-',label=r'${Pr} = 1$')
plot(w_rms_ra20pr8[31:],z[31:],'k-.',label=r'${Pr} = \infty$')

xlabel(r'$[w^\prime]_{RMS}$',fontsize=11)
ylabel(r'$Z$',fontsize=11)

legend(loc=3,framealpha=0)

savefig('figures/w_rms.png',dpi=100,bbox_inches='tight')

fig, ax = subplots(figsize=(pi,pi))

xlim(0,10)
ylim(0.5,1.0)

plot(w_rms_ra20pr1[31:]/u_rms_ra20pr1[31:],z[31:])
savefig('figures/wu_rms.png',dpi=100,bbox_inches='tight')

fig, ax = subplots(figsize=(pi,pi))

xlim(0,2)
ylim(0.5,1.0)

plot(t_rms_ra20pr1[31:],z[31:],'k-',label=r'${Pr} = 1$')
plot(t_rms_ra20pr8[31:],z[31:],'k-.',label=r'${Pr} = \infty$')

xlabel(r'$[\theta^\prime]_{RMS}$',fontsize=11)
ylabel(r'$Z$',fontsize=11)

legend(loc=6,framealpha=0)
savefig('figures/t_rms.png',dpi=100,bbox_inches='tight')

fig, ax = subplots(figsize=(pi,pi))

xlim(0,1)
ylim(0,1)

# plot(z,z)y_int = integrate.cumtrapz(y, x, initial=1)


plot(z,1-z,'k-')
plot(cumtrapz(theta_ra20pr1,z,initial=0) + 1,z,'k-',label=r'${Pr} = 1$')
plot(cumtrapz(theta_ra20pr8,z,initial=0) + 1,z,'k-.',label=r'${Pr} = \infty$')

xlabel(r'$\langle \overline{\theta} \rangle_t$',fontsize=11)
ylabel(r'$Z$',fontsize=11)

legend(loc=3,framealpha=0)
savefig('figures/theta.png',dpi=100,bbox_inches='tight')

fig, ax = subplots(figsize=(pi,pi))

xlim(0,2)
ylim(0.5,1.0)

plot(w_rms_ra20pr1[31:]/u_rms_ra20pr1[31:],z[31:],'k-',label=r'${\widetilde{Ra}} = 20$')
plot(w_rms_ra80pr1[47:]/u_rms_ra80pr1[47:],z_80[47:],'k-.',label=r'${\widetilde{Ra}} = 80$')

xlabel(r'$[w^\prime]_{RMS}/[u^\prime]_{RMS}$',fontsize=11)
ylabel(r'$Z$',fontsize=11)

legend(loc=0,framealpha=0)
savefig('figures/wu_rms.png',dpi=100,bbox_inches='tight')

fig, ax = subplots(figsize=(pi,pi))

xlim(0,2)
ylim(0.5,1.0)

plot(w_rms_ra20pr1[31:]/v_rms_ra20pr1[31:],z[31:],'k-',label=r'${\widetilde{Ra}} = 20$')
plot(w_rms_ra80pr1[47:]/v_rms_ra80pr1[47:],z_80[47:],'k-.',label=r'${\widetilde{Ra}} = 80$')

xlabel(r'$[w^\prime]_{RMS}/[v^\prime]_{RMS}$',fontsize=11)
ylabel(r'$Z$',fontsize=11)

legend(loc=0,framealpha=0)
savefig('figures/wv_rms.png',dpi=100,bbox_inches='tight')
