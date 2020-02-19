import numpy as np
from dedalus import public as de
from dedalus.extras import flow_tools

import logging
root = logging.root
for h in root.handlers:
    h.setLevel("DEBUG")
logger = logging.getLogger(__name__)

Nx  = 64
Ny  = 64
Nz  = 64

Lc  = 4.8154
Lx  = 20*Lc
Ly  = 20*Lc
Lz  = 1.0

Ra  = 20.0
A   = Lx*Ly

x_basis = de.Fourier('x', Nx, interval=(0.0,Lx), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(0.0,Ly), dealias=3/2)  
z_basis = de.SinCos ('z', Nz, interval=(0.0,Lz), dealias=3/2)

domain  = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)
qg      = de.IVP(domain, variables=['tf','w','si','u','v','zeta'], time = 't')

qg.meta['tf']['z']['parity']    = -1
qg.meta['si']['z']['parity']    = +1
qg.meta['w']['z']['parity']     = -1
qg.meta['u']['z']['parity']     = +1
qg.meta['v']['z']['parity']     = +1
# qg.meta['omega']['z']['parity']     = -1
qg.meta['zeta']['z']['parity']  = +1

qg.parameters['Ra']             = Ra
qg.parameters['Axy']            = A

qg.substitutions['J(A,B)']      = "dx(A)*dy(B) - dy(A)*dx(B)"
qg.substitutions['L(A)']        = "dx(dx(A)) + dy(dy(A))"
qg.substitutions['D(A)']        = "d(A,x=4) + d(A,y = 4)"
qg.substitutions['M(A,B)']        = "- 1 + integ(integ(A*B - integ(A*B,'z'),'y'),'x')/Axy"

# equations 2.29(a-c) + 3.2 [Sprague, 2006]
qg.add_equation("dt(tf) - L(tf) = - J(si,tf) + w*M(w,tf)")

qg.add_equation("dz(w) + D(si) = 0",condition="(nx != 0) and (ny != 0)")
qg.add_equation("w = 0",condition="(nx == 0) or (ny == 0)")

qg.add_equation("dz(si) - Ra*tf - L(w) = 0",condition="(nx != 0) and (ny != 0)")
qg.add_equation("si = 0",condition="(nx == 0) or (ny == 0)")

# diagnostics
qg.add_equation("u + dy(si) = 0")
qg.add_equation("v - dx(si) = 0")
# qg.add_equation("w - L(fi) = 0")
qg.add_equation("zeta - L(si) = 0")

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

CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'v','w'))

snap = solver.evaluator.add_file_handler('inf/snapshots', sim_dt=0.2, max_writes=100)

snap.add_task("interp(zeta, z=0.0)", scales=1, name='w bottom')
snap.add_task("interp(zeta, z=0.5)", scales=1, name='w midplane')
snap.add_task("interp(zeta, z=1.0)", scales=1, name='w top')

snap.add_task("interp(tf, z=0.0)", scales=1, name='f bottom')
snap.add_task("interp(tf, z=0.5)", scales=1, name='f midplane')
snap.add_task("interp(tf, z=1.0)", scales=1, name='f top')

series = solver.evaluator.add_file_handler('inf/series', sim_dt=0.2, max_writes=100)
series.add_task("interp(-M(w,tf), z=1.0)", scales=1, name='Nu')

flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("w*tf - integ(w*tf,'z')", name='Nu')

while solver.ok:

    dt = CFL.compute_dt()
    dt = solver.step(dt)
    
    if(solver.iteration%20 == 0):
        logger.info('Iteration: %i, Time step: %e' %(solver.iteration, dt))
        logger.info('Nu = %f' %flow.volume_average('Nu'))

