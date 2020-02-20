"""
Rotationally constrained thermal convection
in a plane layer with high aspect ratio.

Reduced asymptotic low Rossby problem equations from:
Sprague, Julien, Knobloch and Werne (2006) JFM

Parameters:
    Axy Transverse section area
    Ra  Scaled Rayleigh number
    Pr  Prandtl number

Variables:
    tf  Temperature fluctuation
    tz  Mean vertical temperature gradient
    si   Pressure/streamfunction
    w   Vertical velocity

Diagnostics:
    u,v Velocity
    ze  Vertical vorticity
"""

import numpy as np
from dedalus import public as de
from dedalus.extras import flow_tools

import time
import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")
logger = logging.getLogger(__name__)

import yaml

with open(r'input.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

Nx,Ny   = (params['nx'],params['nx'])
Nz      = params['nz']
Lx,Ly   = (params['lx']*params['lc'],params['lx']*params['lc'])
Lz      = 1.0
Axy     = Lx*Ly
Ra      = params['ra']
Pr      = params['pr']

logger.info('Parameters loaded for case: %s' %(params['case']))

x_basis = de.Fourier('x', Nx, interval=(0.0,Lx), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(0.0,Ly), dealias=3/2)  
z_basis = de.SinCos ('z', Nz, interval=(0.0,Lz), dealias=3/2)
domain  = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

variables                           = ['tf','w','si','u','v','ze','tz']
problem                             = de.IVP(domain, variables=variables, time='t')
problem.meta['tf']['z']['parity']   = -1
problem.meta['w']['z']['parity']    = -1
problem.meta['tz']['z']['parity']   = +1
problem.meta['si']['z']['parity']   = +1
problem.meta['u']['z']['parity']    = +1
problem.meta['v']['z']['parity']    = +1
problem.meta['ze']['z']['parity']   = +1
problem.parameters['Ra']            = Ra
problem.parameters['Axy']           = Axy

problem.substitutions['J(A,B)']     = "dx(A)*dy(B) - dy(A)*dx(B)"
problem.substitutions['L(A)']       = "dx(dx(A)) + dy(dy(A))"
problem.substitutions['D(A)']       = "d(A, x=4) + 2.0*d(A, x=2, y=2) + d(A, y=4)"
problem.substitutions['M(A,B)']     = "(1.0/Axy)*integ(integ((A*B - integ(A*B,'z')), 'y'),'x')"

if Pr == 'inf':

    logger.info('Using infinite Prandtl reduced equations 2.29[abc] + 3.2')

    problem.add_equation("dt(tf) - L(tf)        = - J(si,tf) - w*tz")
    problem.add_equation("tz                    = - 1 + M(w,tf)")
    problem.add_equation("dz(w) + D(si)         = 0", condition="(nx != 0) and (ny != 0)")
    problem.add_equation("w                     = 0", condition="(nx == 0) or (ny == 0)")
    problem.add_equation("dz(si) - Ra*tf - L(w) = 0", condition="(nx != 0) and (ny != 0)")
    problem.add_equation("si                    = 0", condition="(nx == 0) or (ny == 0)")
    problem.add_equation("u + dy(si)            = 0")
    problem.add_equation("v - dx(si)            = 0")
    problem.add_equation("ze - L(si)            = 0")

elif Pr >= 1:

    logger.info('Using finite Prandtl reduced equations 2.27[abc] + 3.1')
    
    problem.parameters['Pr']        = Pr

    problem.add_equation("dt(w) + dz(si) - (Ra/Pr)*tf - L(w)    = J(si,w)",         condition="(nx != 0) and (ny != 0)")
    problem.add_equation("w                                     = 0",               condition="(nx == 0) or (ny == 0)")
    problem.add_equation("dt(L(si)) - dz(w) - D(si)             = - J(si,L(si))",   condition="(nx != 0) and (ny != 0)")
    problem.add_equation("si                                    = 0",               condition="(nx == 0) or (ny == 0)")
    problem.add_equation("dt(tf) - (1.0/Pr)*L(tf)               = - J(si,tf) - w*tz")
    problem.add_equation("tz                                    = - 1 + Pr*M(w,tf)")
    problem.add_equation("u + dy(si)                            = 0")
    problem.add_equation("v - dx(si)                            = 0")
    problem.add_equation("ze - L(si)                            = 0")

else:

    logger.info('That does not make sense!')

ts      = de.timesteppers.RK443
solver  = problem.build_solver(ts)
logger.info('Building solver... success!')

# Initial Condition
gshape  = domain.dist.grid_layout.global_shape(scales=1)
slices  = domain.dist.grid_layout.slices(scales=1)
rand    = np.random.RandomState(seed=42)
noise   = rand.standard_normal(gshape)[slices]
pert    = 1e-1*noise
tf      = solver.state['tf']
tz      = solver.state['tz']
tf['g'] = pert
tz['g'] = -1.0

# tf = solver.state['tf']
# tf['g'] = np.random.rand(*tf['g'].shape)

dt = np.float(params['dt'])

solver.stop_sim_time    = params['st']
solver.stop_wall_time   = params['wt']*60.
solver.stop_iteration   = np.inf

CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.5,
                            max_change=1.25, min_change=0.5, max_dt=0.1, threshold=0.05)
CFL.add_velocities(('u', 'v','w'))

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.05, max_writes=10, mode='overwrite')
snapshots.add_task("interp(ze, z=0.0)", scales=1, name='ze bottom')
snapshots.add_task("interp(w, y=0.5)",  scales=1, name='w vertical')
snapshots.add_task("interp(tf, y=0.5)", scales=1, name='tf vertical')
snapshots.add_task("interp(tf, z=0.0)", scales=1, name='tf bottom')
snapshots.add_task("interp(tf, z=1.0)", scales=1, name='tf top')

profiles = solver.evaluator.add_file_handler('profiles', sim_dt=0.2, max_writes=100)
profiles.add_task("sqrt(integ(integ(tf*tf, 'x'), 'y'))", scales=1, name='tf_rms')
profiles.add_task("sqrt(integ(integ(u*u, 'x'), 'y'))", scales=1, name='u_rms')
profiles.add_task("sqrt(integ(integ(v*v, 'x'), 'y'))", scales=1, name='v_rms')
profiles.add_task("sqrt(integ(integ(w*w, 'x'), 'y'))", scales=1, name='w_rms')
profiles.add_task("sqrt(integ(integ(ze*ze, 'x'), 'y'))", scales=1, name='ze_rms')

series = solver.evaluator.add_file_handler('series', sim_dt=0.2, max_writes=100)
series.add_task("interp(-tz, z=0.5)", scales=1, name='Nu')

flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("interp(-tz,z=1.0)", name='Nu')

try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:

        dt = CFL.compute_dt()
        dt = solver.step(dt)

        if (solver.iteration-1)%10 == 0:
            logger.info('Iteration: %i, Step size: %e, Run time: %f' %(solver.iteration, dt, solver.sim_time))
            logger.info('Nu = %f' %(flow.max('Nu')))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))