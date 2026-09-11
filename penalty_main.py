from penalty_utils import *
from penaltyDE_2d import *
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--lam", type=float, default=6)
parser.add_argument("-N", "--num_grid_points", type=int, default=5)
parser.add_argument("-dt", "--timestep", type=float, default=0.01)
parser.add_argument("-t", "--time", type=float, default=1.)
parser.add_argument("-e", "--eqtype", type=str, default='heat')
parser.add_argument(
    "-b", "--bctype", type=str, default='wall',
    help=(
        "boundary geometry/condition: wall, circle, slit, or neumann "
        "(wall_deriv and wall_neumann are aliases for neumann)"
    ),
)
parser.add_argument("-v", "--bcval", type=str, default='zero')
parser.add_argument("-s", "--simtype", type=str, default='errors')
args = parser.parse_args()
### Setup
N = int(2**(args.num_grid_points))
dt = args.timestep
LAM = 10**(args.lam)
T = args.time
eqtype = args.eqtype.lower()
bctype = args.bctype.lower()
bcval = args.bcval.lower()
simtype = args.simtype.lower()

if bctype in {'neumann', 'wall_neumann', 'wall_deriv'} and bcval != 'zero':
    parser.error('Neumann boundaries currently support only --bcval zero')

plot_errors_heat    = True
do_non_zero_bc_heat = False
wave_stuff_zero     = True
do_source = True


if simtype == 'errors' and eqtype == 'heat':
    plot_errors_heat = True
if bcval == 'nonzero' and eqtype == 'heat':
    do_non_zero_bc_heat = True
if simtype == 'errors' and eqtype == 'wave':
    plot_errors_wave = True

print("Running time evolution for {eqt} equation with T={TT} and lambda={lala:E}.".format(eqt=eqtype, TT=T, lala=LAM))

save_figs = True

''' heat start '''
if eqtype.lower() == 'heat':
    kappa = 4.
    HeatDE = PenaltyDE_2d(N=N, dt=dt, T=T, lam=LAM, eqtype=eqtype, coeff=kappa, bctype=bctype)
    HeatDE.construct_initial_state_centered(width=1/(3/2*HeatDE.N), height=1)
    # If non-zero boundary conditions
    if do_non_zero_bc_heat:
        g_fn = (HeatDE.x*(HeatDE.x<=1/2) + (1-HeatDE.x)*(HeatDE.x>1/2))
        HeatDE.set_bcvals(in_vals=(1-HeatDE.y)*g_fn)
    # If source term
    if do_source:
        src_RHS = np.zeros_like(HeatDE.x, dtype=complex).flatten() # set point source in middle equal to IC value
        points = circle_points(HeatDE.x, HeatDE.y, radius=0.1, kind='inside')
        src_RHS[points] = 298
        HeatDE.set_RHS(in_vals=src_RHS)
    ''' -------------------------------------------------- '''

    # Run loop over lambdas and plot error scaling
    if plot_errors_heat:
        lambdas = np.logspace(2, 6, 4)
        errors = np.zeros(len(lambdas))

        # use left-right-back-boundary for error computation if it's the nonzero boundary condition
        wall_lrb = HeatDE.wall_bdry_projection2d(kind='lrb')
        wall_t = HeatDE.wall_bdry_projection2d(kind='t')
        for i_l, lam in enumerate(lambdas):
            HeatDE.lam = lam  # reset to current lambda
            vals, bndry_vals = HeatDE.time_evolution()
            if HeatDE.projc.kind == 'value':
                errors[i_l] = np.linalg.norm(np.real(bndry_vals[-1,:]))
            elif HeatDE.projc.kind == 'deriv':
                print('curr err', np.linalg.norm(HeatDE.projc.operator.dot(bndry_vals[-1,:].T).T))
                errors[i_l] = np.linalg.norm(HeatDE.projc.operator.dot(bndry_vals[-1,:].T).T)
            # After errors are computed we can add the boundary values to the vals array
            # but we do not need to do that here if we are interested only in the errors

        onorm = sparse.linalg.norm(HeatDE.operator)
        x0norm = np.linalg.norm(HeatDE.x0)
        rhsnorm = np.linalg.norm(HeatDE.RHS)  # here RHS is constant so we can just pick something
        # print('Operator norm', onorm)
        # print('initial value norm', x0norm)

        err_bounds = 2*np.sqrt(onorm *
                (x0norm**2 + 2*T*x0norm*rhsnorm + T**2 * rhsnorm**2) *
                np.power(lambdas, -1))

        print('Errors', errors)
        print('Error bounds', err_bounds)
        plt.loglog(lambdas, errors, 'cd-', label=r'$\|\mathbf{u}\|_{S_c}$')
        plt.loglog(lambdas, err_bounds, 'r-.', label=r'upper bound')
        data = pd.DataFrame({'lambdas': lambdas, 'errors': errors, 'bounds': err_bounds,
            'N':N, 'dt':dt, 'bctype':bctype})
        data.to_csv('errors_t{TT}_{ET}_N{NN}_dt{DD}_type{simt}_btype{bct}_coeff{cc}_bcval{bcv}.csv'.format(TT=T, ET=eqtype, simt=simtype, NN=N, DD=dt, bct=bctype,cc=kappa,bcv=bcval))
        plt.legend()
        plt.xlabel(r'$\lambda$')
        plt.ylabel('error')
        if save_figs:
            plt.savefig('errors_t{TT}_{ET}_N{NN}_dt{DD}_type{simt}_btype{bct}_coeff{cc}_bcval{bcv}.pdf'.format(TT=T, ET=eqtype, simt=simtype, NN=N, DD=dt, bct=bctype,cc=kappa,bcv=bcval))
        else:
            plt.show()
        HeatDE.lam = LAM
        vals, bndry_vals = HeatDE.time_evolution()
        ''' plot final '''
        bndry_vals = bndry_vals.reshape((-1,HeatDE.N,HeatDE.N))
        # min and max vals for colorbar
        minval, maxval = np.min(np.real(vals.flatten())), np.real(np.max(vals.flatten()))
        minb, maxb = np.min(np.real(bndry_vals.flatten())), np.max(np.real(bndry_vals.flatten()))
        # plot initial and final results
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, layout='constrained', figsize=(12,12))
        cax0 = ax[0,0].imshow(np.real(vals[0,:]), cmap='cividis', vmin=minval, vmax=maxval)
        cax1 = ax[1,0].imshow(np.real(bndry_vals[0,:]), cmap='viridis', vmin=minb, vmax=maxb)
        cax0 = ax[0,1].imshow(np.real(vals[-1,:]), cmap='cividis', vmin=minval, vmax=maxval)
        cax1 = ax[1,1].imshow(np.real(bndry_vals[-1,:]), cmap='viridis', vmin=minb, vmax=maxb)
        fig.colorbar(cax0, ax=[ax[0,0], ax[0,1]], location='bottom', shrink=.4)
        fig.colorbar(cax1, ax=[ax[1,0], ax[1,1]], location='bottom', shrink=.4)
        fig.suptitle(r'\bfseries Heat equation with nonzero boundary and $\lambda=10^{LL}$'.format(LL=args.lam))
        ax[0,0].set_title(r'Solution at $T=0$')
        ax[1,0].set_title(r'Boundary errors at $T=0$')
        ax[0,1].set_title(r'Solution at $T={TT:.1f}$'.format(TT=T))
        ax[1,1].set_title(r'Boundary errors at $T={TT:.1f}$'.format(TT=T))
        data = pd.DataFrame({'lambda': HeatDE.lam, 'vals':vals.flatten(), 'bvals':bndry_vals.flatten(),
            'N':N, 'dt':dt, 'bctype':bctype})
        data.to_csv('Heat_{bqt}-boundary_lambda{LL}_{DD}_{NN}_kap{cc}_bcval{bcv}.csv'.format(bqt=HeatDE.bctype,
                LL=HeatDE.lam,DD=HeatDE.dt,NN=HeatDE.N,cc=kappa,bcv=bcval))
        if save_figs:
            fig.savefig('Heat_{bqt}-boundary_lambda{LL}_{DD}_{NN}_{cc}_bcval{bcv}.pdf'.format(bqt=HeatDE.bctype,
                LL=HeatDE.lam,DD=HeatDE.dt,NN=HeatDE.N,cc=kappa,bcv=bcval))
        else:
            plt.show()

    ''' -------------------------------------------------- '''
    plot_evolution = False
    plot_bdry_evolution = False
    # Plot evolution
    if plot_evolution:
        HeatDE.lam = LAM
        val_array, bndry_vals = HeatDE.time_evolution()
        if do_non_zero_bc_heat:
            print('am here')
            val_array += np.reshape(HeatDE.wall_bdry_projection2d()@HeatDE.bcvals, (HeatDE.N,HeatDE.N))  # still cheaping out given we have constant RHS/BC
        do_plot_evolution(val_array, HeatDE.dt)
        if plot_bdry_evolution:
            #PLOT ANIMATED BOUNDARY VALUES OVER TIME
            bndry_vals = np.reshape(bndry_vals, (-1,HeatDE.N,HeatDE.N))
            do_plot_evolution(bndry_vals, HeatDE.dt)
    ''' heat end '''
''' wave start '''
if eqtype == 'wave':
    c_squared = (1., 1.)
    print('cfl is ', np.max(c_squared)*dt*N)
    kwargs = {'kind': 'lrtb'}
    WaveDE = PenaltyDE_2d(N=N, dt=dt, T=T, lam=LAM, eqtype=eqtype, coeff=c_squared, bctype=bctype, **kwargs)
    WaveDE.construct_initial_state_wave()
    # Should be induced-2-norm and not fro but fro >= 2 and that's more reliably computed
    bound = sparse.linalg.norm(WaveDE.operator, ord='fro') * np.linalg.norm(np.real(WaveDE.x0))**2
    bound /= WaveDE.lam
    bound = bound**(1/2)

    plot_errors_wave = True
    if plot_errors_wave:
        lambdas = np.logspace(2, 6, 4)
        errors = np.zeros(len(lambdas))
        for i_l, lam in enumerate(lambdas):
            WaveDE.lam = lam  # reset to current lambda
            vals, bndry_vals = WaveDE.time_evolution()
            # do_plot_evolution_wave(vals, WaveDE.dt)
            # bndry_vals stores P_c v.  As in the paper, report the
            # constraint violation for the physical field u only.
            errors[i_l] = np.linalg.norm(bndry_vals[-1, 0, :])
            # After errors are computed we can add the boundary values to the vals array
            # but we do not need to do that here if we are interested only in the errors
        onorm = sparse.linalg.norm(WaveDE.operator)
        u_proj = sparse.block_array([[identity(WaveDE.N**2), None],[None,sparse.diags_array(np.zeros(WaveDE.N**2,dtype='complex'))]])
        x0norm = np.linalg.norm(u_proj.dot(WaveDE.x0))
        rhsnorm = np.linalg.norm(WaveDE.RHS)  # here RHS is constant so we can just pick something

        err_bounds = 2*np.sqrt(onorm *
                (x0norm**2 + 2*WaveDE.T*x0norm*rhsnorm + WaveDE.T**2 * rhsnorm**2) *
                np.power(lambdas, -1))
        print('Errors', errors)
        print('Error bounds', err_bounds)

        data = pd.DataFrame({'lambdas': lambdas, 'errors': errors, 'bounds': err_bounds,
            'N':N, 'dt':dt, 'bctype':bctype})
        data.to_csv('wave_errors_t{TT}_N{NN}_dt{DD}_type{simt}_btype{bct}.csv'.format(TT=T, simt=simtype, NN=N, DD=dt, bct=bctype))
        plt.loglog(lambdas, errors, 'cd-', label=r'$\|\mathbf{u}\|_{S_c}$')
        plt.loglog(lambdas, err_bounds, 'r-.', label=r'upper bound')
        plt.legend()
        plt.xlabel(r'$\lambda$')
        plt.ylabel('error')
        plt.tight_layout()
        if save_figs:
            plt.savefig('wave_errors_t{TT}_N{NN}_dt{DD}_type{simt}_btype{bct}.pdf'.format(TT=T, simt=simtype, NN=N, DD=dt, bct=bctype))
        else:
            plt.show()
    # If plot time evolution for wave equation
    do_wave_figure = True
    if do_wave_figure:

        WaveDE.lam = LAM
        vals, bndry_vals = WaveDE.time_evolution()
        data = pd.DataFrame({'lambda': WaveDE.lam, 'vals':vals.flatten(), 'bvals':bndry_vals.flatten(),
            'N':N, 'dt':dt, 'bctype':bctype})
        data.to_csv('Wave_{bqt}-boundary_lambda{LL}_{DD}_{NN}.csv'.format(bqt=WaveDE.bctype,
                LL=WaveDE.lam,DD=WaveDE.dt,NN=WaveDE.N))
        # plot final
        bndry_vals = bndry_vals.reshape((-1,2,WaveDE.N,WaveDE.N))
        # min and max vals for colorbar
        minu, maxu = np.min(np.real(vals[:,0,...].flatten())), np.real(np.max(vals[:,0,...].flatten()))
        minw, maxw = np.min(np.real(vals[:,1,...].flatten())), np.real(np.max(vals[:,1,...].flatten()))
        minb, maxb = np.min(np.real(bndry_vals.flatten())), np.max(np.real(bndry_vals.flatten()))
        # plot initial and final results
        fig, ax = plt.subplots(nrows=2, ncols=2, layout='constrained', figsize=(12,12))
        # layout: u-0    u-T
        #         w-0    w-T
        # deleted #       err-0  err-T
        cax0 = ax[0,0].imshow(np.real(vals[0,0,:]), cmap='cividis', vmin=minu, vmax=maxu)
        cax0 = ax[0,1].imshow(np.real(vals[-1,0,:]), cmap='cividis', vmin=minu, vmax=maxu)
        cax1 = ax[1,0].imshow(np.real(vals[0,1,:]), cmap='cividis', vmin=minw, vmax=maxw)
        cax1 = ax[1,1].imshow(np.real(vals[-1,1,:]), cmap='cividis', vmin=minw, vmax=maxw)
        fig.colorbar(cax0, ax=[ax[0,0], ax[0,1]], location='bottom', shrink=.4)
        fig.colorbar(cax1, ax=[ax[1,0], ax[1,1]], location='bottom', shrink=.4)
        fig.suptitle(r'Wave equation with zero boundary and $\lambda=10^{LL}$'.format(LL=int(args.lam)))
        ax[0,0].set_title(r'$\mathbf{u}(0)$')
        ax[0,1].set_title(r'$\mathbf{u}$'+r'$(T={TT:.1f})$'.format(TT=T))
        ax[1,0].set_title(r'$\mathbf{w}(0)$')
        ax[1,1].set_title(r'$\mathbf{w}$' +r'$(T={TT:.1f})$'.format(TT=T))
        if save_figs:
            fig.savefig('Wave_{bqt}-boundary_lambda{LL}_{DD}_{NN}.pdf'.format(bqt=WaveDE.bctype,
                LL=WaveDE.lam,DD=WaveDE.dt,NN=WaveDE.N))
        else:
            plt.show()
