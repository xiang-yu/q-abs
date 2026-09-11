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
    return projOp(N*N, slit_indices(X, Y))

''' -------------------------------------------------- '''
def wall_neumann_pairs(num_grid_points, X, Y, kind='lrtb'):
    '''
    Return disjoint ``(boundary, inward-neighbour)`` pairs for a wall.

    A two-point zero-Neumann stencil requires both values in a pair to agree.
    The pairs must be disjoint for their simultaneous swaps to be an
    involution.  Corners are omitted because their inward normal is
    ambiguous; when two side stencils would share an inward point, the pair
    from the later side is omitted as required by Eqs. (105)--(106) of the
    accompanying paper.
    '''
    N = num_grid_points
    requested_sides = set(kind.lower())
    invalid_sides = requested_sides.difference('lrtb')
    if invalid_sides or not requested_sides:
        raise ValueError("kind must be a non-empty subset of 'lrtb'")
    if X.shape != (N, N) or Y.shape != (N, N):
        raise ValueError('X and Y must both have shape (num_grid_points, num_grid_points)')

    corners = {0, N - 1, N * (N - 1), N * N - 1}
    inward_offsets = {'t': N, 'b': -N, 'l': 1, 'r': -1}
    used_points = set()
    pairs = []

    # Top and bottom first reproduces the stencil layout used in the paper:
    # side pairs adjacent to them are skipped when their inner points clash.
    for side in 'tblr':
        if side not in requested_sides:
            continue
        for boundary in wall_indices(X, Y, L=0., R=1., kind=side):
            boundary = int(boundary)
            neighbour = boundary + inward_offsets[side]
            if boundary in corners:
                continue
            if boundary in used_points or neighbour in used_points:
                continue
            pairs.append((boundary, neighbour))
            used_points.update((boundary, neighbour))

    return np.asarray(pairs, dtype=int).reshape((-1, 2))


def wall_deriv_projection2d(num_grid_points, X, Y, kind='lrtb'):
    '''
    Construct the orthogonal projector for a zero-Neumann wall condition.

    If ``S`` simultaneously swaps every disjoint boundary/inner pair, the
    infeasible (unequal-value) subspace is projected onto by
    ``P_N = (I - S) / 2``.  Building the two-by-two projector blocks directly
    leaves all unaffected grid points in the kernel and avoids materialising
    the full swap.
    '''
    N = num_grid_points
    pairs = wall_neumann_pairs(N, X, Y, kind=kind)
    if not len(pairs):
        return sparse.csc_matrix((N * N, N * N), dtype=float)

    first, second = pairs.T
    rows = np.concatenate((first, first, second, second))
    cols = np.concatenate((first, second, first, second))
    data = np.concatenate((
        0.5 * np.ones(len(pairs)),
        -0.5 * np.ones(len(pairs)),
        -0.5 * np.ones(len(pairs)),
        0.5 * np.ones(len(pairs)),
    ))
    return sparse.coo_matrix(
        (data, (rows, cols)), shape=(N * N, N * N)
    ).tocsc()
''' -------------------------------------------------- '''
def ipic_operator(operator, projc, lam):
    '''Return the interaction-picture generator for a fixed projector.

    The four time-independent sparse blocks are computed once.  Each call
    then only applies scalar phases, rather than forming two matrix
    exponentials and carrying out two sparse matrix products.
    '''
    operator = sparse.csc_matrix(operator, dtype=complex)
    P = projc.operator
    Q = projc.complement

    QAQ = (Q @ operator @ Q).tocsc()
    PAP = (P @ operator @ P).tocsc()
    PAQ = (P @ operator @ Q).tocsc()
    QAP = (Q @ operator @ P).tocsc()

    def wrap_ipic(t):
        phase = np.exp(1.j * lam * t)
        inverse_phase = np.exp(-1.j * lam * t)
        return QAQ + PAP + phase * PAQ + inverse_phase * QAP
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
        fig.suptitle(f'time_step {frame*dt:.2f}')
        ax[0].set_title(r'$v$')
        ax[1].set_title(r'$w=\partial_t v$')
        return [cax0,cax1]
    n_steps = val_array.shape[0]//100
    anim = FuncAnimation(fig, update, frames=range(0, val_array.shape[0], n_steps), interval=0, blit=False)
    plt.show()
