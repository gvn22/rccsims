import numpy as np
from dedalus import public as de
from dedalus.extras import flow_tools

import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")
logger = logging.getLogger(__name__)

Nx  = 64
Ny  = 64
Nz  = 64

Lc  = 4.8154
Lx  = 10*Lc
Ly  = 10*Lc
Lz  = 1.0

x_basis = de.Fourier('x', Nx, interval=(0.0,Lx), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(0.0,Ly), dealias=3/2)  
z_basis = de.SinCos ('z', Nz, interval=(0.0,Lz), dealias=3/2)

domain  = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)
qg = de.IVP(domain, variables=['tf','fi','si','u','v','w'], time = 't')

qg.meta['tf']['z']['parity']    = -1
qg.meta['si']['z']['parity']    = +1
qg.meta['fi']['z']['parity']    = -1
qg.meta['u']['z']['parity']     = +1
qg.meta['v']['z']['parity']     = +1
qg.meta['w']['z']['parity']     = -1

qg.parameters['Ra']             = 20.0
qg.parameters['Pr']             = 1.0
qg.parameters['Axy']            = Lx*Ly

qg.substitutions['J(A,B)']      = "dx(A)*dy(B) - dy(A)*dx(B)"
qg.substitutions['L(A)']        = "dx(dx(A)) + dy(dy(A))"
qg.substitutions['D(A)']        = "L(L(A))"
qg.substitutions['M(A)']        = "- 1 + Pr*integ(integ(A - integ(A,'z'),'y'),'x')/Axy"

# equations 2.27(a-c) + 3.1 [Sprague, 2006]
qg.add_equation("dt(tf) - L(tf)/Pr = - J(si,tf) + L(fi)*M(fi*tf)")

qg.add_equation("dt(L(si)) - dz(L(fi)) - D(si) = J(si,L(si))",condition="(nx != 0) and (ny != 0)")
qg.add_equation("fi = 0",condition="(nx == 0) or (ny == 0)")

qg.add_equation("dt(L(fi)) + dz(si) - Ra*tf - D(fi) = J(si,L(fi))",condition="(nx != 0) and (ny != 0)")
qg.add_equation("si = 0",condition="(nx == 0) or (ny == 0)")

# auxilliary equations
qg.add_equation("u + dy(si) = 0")
qg.add_equation("v - dx(si) = 0")
qg.add_equation("w - L(fi) = 0")

ts      = de.timesteppers.RK443
solver  = qg.build_solver(ts)
logger.info('Building solver... success!')

# random temperature anomalies
tf = solver.state['tf']
tf['g'] = np.random.rand(*tf['g'].shape)

dt = 0.01

solver.stop_sim_time = 100
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

u = solver.state['u']
v = solver.state['v']
w = solver.state['w']

CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'v','w'))

snap = solver.evaluator.add_file_handler('snapshots', sim_dt=0.2, max_writes=10)

# vorticity
snap.add_task("interp(si, z=0.0)", scales=1, name='w bottom')
snap.add_task("interp(si, z=0.5)", scales=1, name='w midplane')
snap.add_task("interp(si, z=1.0)", scales=1, name='w top')

# temperature anomaly
snap.add_task("interp(tf, z=0.0)", scales=1, name='f bottom')
snap.add_task("interp(tf, z=0.5)", scales=1, name='f midplane')
snap.add_task("interp(tf, z=1.0)", scales=1, name='f top')

while solver.ok:

    if(solver.iteration%20 == 0):
        logger.info('Iteration: %i, Mean: %e' %(solver.iteration, np.mean(tf['g']**2)))
    dt = CFL.compute_dt()
    dt = solver.step(dt)