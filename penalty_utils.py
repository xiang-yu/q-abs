import findiff
from findiff import Diff
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as mticker

from scipy import sparse
from scipy.sparse import identity, kron, diags
import numpy as np
# from helpers import *
from scipy.integrate import RK45, LSODA, BDF, RK23

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "mathptmx",
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.size": 20
})

''' -------------------------------------------------- '''

def projOp(N, indices):
    # this function is adapted Tyler's block encoding repository
    # https://github.com/Kharazitd/BlockEncodingDemos
    assert np.max(indices) < N
    diagVals = np.zeros(N)
    diagVals[indices] = 1
    proj = diags(diagVals, offsets = 0, format='csc')
    return proj

''' -------------------------------------------------- '''

def get_nonzero_points(condition):
    return np.flatnonzero(condition.flatten())

''' -------------------------------------------------- '''

def circle_points(X, Y, center=(0.5, 0.5), radius=0.5, kind='outside'):
    if kind == 'outside':
        condition = (X-center[0])**2 + (Y-center[1])**2 > radius**2
    elif kind == 'inside':
        condition = (X-center[0])**2 + (Y-center[1])**2 <= radius**2
    else:
        raise Exception("Provide either kind='outside' or kind='inside'.")
    '''
    condition = condition.flatten()
    points = np.nonzero(condition)
    '''
    return get_nonzero_points(condition)

''' -------------------------------------------------- '''

def wall_indices(X, Y, L=0., R=1., kind='lrtb'):
    conditions = np.zeros_like(X, dtype=bool)
    if 'l' in kind.lower():
        conditions += np.isclose(X, L, atol=1/2*(X[0,1]-X[0,0]))
    if 'r' in kind.lower():
        conditions += np.isclose(X, R, atol=1/2*(X[0,1]-X[0,0]))
    if 't' in kind.lower():
        conditions += np.isclose(Y, L, atol=1/2*(Y[1,0]-Y[0,0]))
    if 'b' in kind.lower():
        conditions += np.isclose(Y, R, atol=1/2*(Y[1,0]-Y[0,0]))

    return get_nonzero_points(conditions)

''' -------------------------------------------------- '''

# A hardcoded slit geometry assuming we have a [0,1]^2 box
def slit_indices(X, Y):
    condition =  (X > 0.25)*(X < 0.45)*(Y > 0.3)*(Y <  0.35)
    condition += (X > 0.55)*(X < 0.75)*(Y > 0.3)*(Y <  0.35)

    return get_nonzero_points(condition)

''' -------------------------------------------------- '''

def heat_operator2d(num_grid_points, dissipativity=(1.,1.), dx=1.):
    N = num_grid_points
    # laplacian (scale-free) with a transport term as well
    if isinstance(dissipativity, float) or isinstance(dissipativity, int):
        dissipativity = (dissipativity, dissipativity)
    transport = (0,0)
    diff_op = dissipativity[0]*Diff(axis=0, grid=dx, periodic=True, acc=4)**2 + dissipativity[1]*Diff(axis=1, grid=dx, periodic=True, acc=4)**2 + transport[0]*Diff(axis=0, grid=dx, periodic=True, acc=4) + transport[1]*Diff(axis=1, grid=dx, periodic=True, acc=4)
    ''' let's make that unperiodic but unconstrained '''

    return diff_op.matrix((N,N))

def wave_operator2d(num_grid_points, speed_of_sound=(1,1), dx=1):
    '''
    parameters: num_gridpoints
                speed_of_sound -- speed of sound in the medium
    Here the solution is
         d/ [v] = [0   1][v]
         dt [w] = [cL  0][w]
    with Laplacian L and c~speed_of_sound (as a diag matrix ~ tuple).
    Often, c~c**2 instead.
    Then, the entry v will be the sought-after variable.
    '''
    if isinstance(speed_of_sound, int):
        speed_of_sound = (speed_of_sound, speed_of_sound)
    N = num_grid_points
    if isinstance(speed_of_sound, float):
        speed_of_sound = (speed_of_sound, speed_of_sound)
    laplace = speed_of_sound[0]*Diff(axis=0, grid=dx, periodic=True, acc=4)**2 + speed_of_sound[1]*Diff(axis=1, grid=dx, periodic=True, acc=4)**2
    lapmat = laplace.matrix((N,N))
    Wave = sparse.block_array([[None, identity(N*N)], [lapmat, None]])

    return Wave

''' -------------------------------------------------- '''

def wall_bdry_projection2d(num_grid_points, X, Y, kind='lrtb'):
    N = num_grid_points
    # project onto boundary
    '''
    projector1d = projOp(N,[0,N-1])
    projector2d = kron(projector1d,identity(N)) + kron(identity(N),projector1d)
    proj_corners = projOp(N**2, [0, N-1, N**2 - N , N**2 -1])
    projector2d -= proj_corners
    return projector2d
    '''
    return projOp(N*N, wall_indices(X=X, Y=Y, kind=kind))

''' -------------------------------------------------- '''

def circle_bdry_projection_2d(num_grid_points, X, Y):
    N = num_grid_points
    return projOp(N*N, circle_points(X, Y, kind='outside'))  # projecting onto the outside of the circle

''' -------------------------------------------------- '''

def slit_bdry_projection2d(num_grid_points, X, Y):
    N = num_grid_points
    # project onto boundary
    '''
    projector2d =  projOp(N*N, list(range(N//2*N+N//4,  N//2*N+N//3))+list(range((1+N//2)*N+N//3  ,(1+N//2)*N+N//3)) )
    projector2d += projOp(N*N, list(range(N//2*N+3*N//8,N//2*N+N//2))+list(range((1+N//2)*N+3*N//8,(1+N//2)*N+N//2)) )
    '''

    return projOp(N*N, slit_indices(X, Y))

''' -------------------------------------------------- '''

def wall_deriv_projection2d(num_grid_points, X, Y):
    def clean(wall):
        wall = np.delete(wall, wall.argmax())
        wall = np.delete(wall, wall.argmin())
        return wall

    N = num_grid_points
    indices = np.arange(0, N*N)

    wall_top = clean(wall_indices(X, Y, L=0., R=1., kind='t'))
    wall_bot = clean(wall_indices(X, Y, L=0., R=1., kind='b'))
    # double-clean two of the sides to avoid doubly applying it in the corner points
    wall_lef = clean(clean(wall_indices(X, Y, L=0., R=1., kind='l')))
    wall_rig = clean(clean(wall_indices(X, Y, L=0., R=1., kind='r')))

    ''' swap part '''
    swap =  sparse.coo_matrix((np.ones_like(wall_top), (wall_top, wall_top+N)), shape=(N*N,N*N))
    swap += sparse.coo_matrix((np.ones_like(wall_bot), (wall_bot, wall_bot-N)), shape=(N*N,N*N))
    swap += sparse.coo_matrix((np.ones_like(wall_lef), (wall_lef, wall_lef+1)), shape=(N*N,N*N))
    swap += sparse.coo_matrix((np.ones_like(wall_rig), (wall_rig, wall_rig-1)), shape=(N*N,N*N))
    swap += swap.T
    # ''' identity part '''
    # swap += sparse.coo_matrix((np.ones_like(wall_top), (wall_top, wall_top)), shape=(N*N,N*N))
    # swap += sparse.coo_matrix((np.ones_like(wall_bot), (wall_bot, wall_bot)), shape=(N*N,N*N))
    # swap += sparse.coo_matrix((np.ones_like(wall_lef), (wall_lef, wall_lef)), shape=(N*N,N*N))
    # swap += sparse.coo_matrix((np.ones_like(wall_rig), (wall_rig, wall_rig)), shape=(N*N,N*N))
    proj = swap

    return proj
# ''' -------------------------------------------------- '''
# def ipic_rotation(projc, lam, kind='value'):
#     def wrap_ipic_rot(t):
#         ''' this will only work for Dirichlet as for Neumann
#             the projection is no diagonal anymore
#         '''
#         return diags(np.exp(1.j*lam*t*projc.diagonal()),0)
#     def wrap_ipic_rot_deriv(t):
#         ''' more expensive but correct; should be possible to diagonalize this one easily though '''
#         return sparse.linalg.expm(1.j*lam*t*projc)
#
#     if kind.lower() == 'value' or not kind.lower:
#         return wrap_ipic_rot
#     elif kind.lower() == 'deriv':
#         return wrap_ipic_rot_deriv
''' -------------------------------------------------- '''
def ipic_operator(operator, projc, lam):
    U = projc.apply_exp_fn(lam)  # this is a f(t)
    def wrap_ipic(t):
        # rotation matrix
        return U(t) @ operator @ U(-t)
    return wrap_ipic
''' -------------------------------------------------- '''
def ode_fn(operator, RHS=lambda t : 0):
    def wrap_ode(t, x):
        return operator(t)@x.flatten() + RHS(t)
    return wrap_ode
''' -------------------------------------------------- '''
def do_plot_evolution(invals, dt):
    # PLOT ANIMATED SOLUTION OVER TIME
    val_array = np.real(invals)
    fig, ax = plt.subplots()
    cax = ax.imshow(np.real(val_array[0]), cmap='viridis',
            vmin=np.min(val_array.flatten()), vmax=np.max(val_array.flatten()))
    fig.colorbar(cax,)
    def update(frame):
        cax.set_array(np.real(val_array[frame,:]))
        ax.set_title(f'time_step {frame*dt:.2f}')
        return [cax]

    n_steps = val_array.shape[0] // 100
    anim = FuncAnimation(fig, update, frames=range(0,val_array.shape[0], n_steps), interval=0, blit=False)
    plt.show()
''' -------------------------------------------------- '''
def do_plot_evolution_wave(invals, dt):
    val_array = np.real(invals)
    fig, ax = plt.subplots(ncols=2)
    cax0 = ax[0].imshow(np.real(val_array[0,0,:]), cmap='viridis',
            vmin=np.min(val_array[:,0,:].flatten()), vmax=np.max(val_array[:,0,:].flatten()))
    cax1 = ax[1].imshow(np.real(val_array[0,1,:]), cmap='viridis',
            vmin=np.min(val_array[:,1,:].flatten()), vmax=np.max(val_array[:,1,:].flatten()))
    fig.colorbar(cax0,)
    fig.colorbar(cax1,)
    def update(frame):
        cax0.set_array(np.real(val_array[frame,0,:]))
        cax1.set_array(np.real(val_array[frame,1,:]))
        # err = np.linalg.norm(np.real(bndry_vals[frame,:]) - g_fn.reshape(bndry_vals[frame,:].shape))
        fig.suptitle(f'time_step {frame*dt:.2f}')
        ax[0].set_title(r'$v$')
        ax[1].set_title(r'$w=\partial_t v$')
        return [cax0,cax1]

    n_steps = val_array.shape[0]//100
    anim = FuncAnimation(fig, update, frames=range(0, val_array.shape[0], n_steps), interval=0, blit=False)
    plt.show()
