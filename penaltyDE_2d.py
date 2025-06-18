from penalty_utils import *
import copy

from scipy.fftpack import dct, idct, dst, idst


class PenaltyProj:
    def __init__(self, operator, kind):
        self.operator = operator
        self.kind = kind
        assert (self.kind.lower() == 'value' or
                self.kind.lower() == 'deriv')
        if self.kind.lower() == 'deriv':
            self.operator = self.operator.tocsc()

    def apply_exp_fn(self, lam):
        if self.kind.lower() == 'value':
            out_fn = lambda t: diags(np.exp(1.j*lam*t*self.operator.diagonal()),0)
        elif self.kind.lower() == 'deriv':
            out_fn = lambda t: sparse.linalg.expm(1.j*lam*t*self.operator)
        return out_fn
""" ========================================== """
class PenaltyDE_2d:
    '''
        todo: docstring
    '''
    def __init__(self, N=int(2**5), dt=1e-2, T=1, lam=1e6, eqtype='heat', coeff=(1,1), bctype='wall', **kwargs):
        self.N = N
        self.dt = dt
        self.T = T
        self.lam = lam
        self.x, self.y, self.dx = self._create_xy()

        self.eqtype = eqtype
        self.bctype = bctype
        self.coeff = coeff
        self._set_system_operator(coeff=coeff)

        self.x0 = None
        self.bndry_points = None # , bndry_vals = None, None

        self.projc = None
        self._set_projector(bctype, **kwargs)

        if not eqtype.lower() == 'wave':
            self.RHS         = np.zeros((self.N*self.N), dtype=complex)
            self.bcvals      = np.zeros((self.N*self.N), dtype=complex)
            self.L_at_bcvals = np.zeros((self.N*self.N), dtype=complex)
        elif eqtype.lower() == 'wave':
            self.RHS         = np.zeros((2*self.N*self.N), dtype=complex)
            self.bcvals      = np.zeros((2*self.N*self.N), dtype=complex)
            self.L_at_bcvals = np.zeros((2*self.N*self.N), dtype=complex)
    ''' -------------------------------------------------- '''
    # construct geometry
    def _create_xy(self):
        ''' for simplicity regular uniform grid in 2d '''
        grid_1d = np.linspace(0, 1, self.N, endpoint=True)
        x, y = np.meshgrid(grid_1d, grid_1d)
        dx  = grid_1d[1] - grid_1d[0]

        return x, y, dx
    ''' -------------------------------------------------- '''
    def wall_bdry_projection2d(self, kind='lrtb'):
        ''' project onto boundary '''
        proj = projOp(self.N*self.N, wall_indices(X=self.x, Y=self.y, kind=kind))
        return proj, proj.diagonal().nonzero()
    ''' -------------------------------------------------- '''
    def circle_bdry_projection2d(self):
        ''' project onto circle boundary '''
        proj =  projOp(self.N*self.N, circle_points(self.x, self.y, kind='outside'))  # projecting onto the outside of the circle
        return proj, proj.diagonal().nonzero()
    ''' -------------------------------------------------- '''
    def slit_bdry_projection2d(self):
        ''' project onto slit boundary '''
        proj =  projOp(self.N*self.N, slit_indices(self.x, self.y))
        return proj, proj.diagonal().nonzero()
    ''' -------------------------------------------------- '''
    def wall_deriv_projection2d(self):
        ''' construct derivative projection based on first-order finite difference stencil '''
        def clean(wall):
            wall = np.delete(wall, wall.argmax())
            wall = np.delete(wall, wall.argmin())
            return wall

        ''' geometry '''
        wall_top = clean(wall_indices(self.x, self.y, L=0., R=1., kind='t'))
        wall_bot = clean(wall_indices(self.x, self.y, L=0., R=1., kind='b'))
        # double-clean two of the sides to avoid doubly applying it in the corner points
        wall_lef = clean(clean(wall_indices(self.x, self.y, L=0., R=1., kind='l')))
        wall_rig = clean(clean(wall_indices(self.x, self.y, L=0., R=1., kind='r')))

        ''' swap part '''
        swap =  sparse.coo_matrix((np.ones_like(wall_top), (wall_top, wall_top+self.N)), shape=(self.N**2,self.N**2))
        swap += sparse.coo_matrix((np.ones_like(wall_bot), (wall_bot, wall_bot-self.N)), shape=(self.N**2,self.N**2))
        swap += sparse.coo_matrix((np.ones_like(wall_lef), (wall_lef, wall_lef+1))     , shape=(self.N**2,self.N**2))
        swap += sparse.coo_matrix((np.ones_like(wall_rig), (wall_rig, wall_rig-1))     , shape=(self.N**2,self.N**2))

        bndry_pts = np.array([*zip(*swap.nonzero())],dtype=int)
        proj = swap + swap.T

        return proj, bndry_pts
    ''' -------------------------------------------------- '''
    def _set_projector(self, bctype, **kwargs):
        '''
            set projection operation depending on boundary condition
            todo for future: mixed conditions with mixed/non-overlapping projections
        '''
        # If heat equation, select one of 'wall', 'slit', 'circle'
        if bctype == 'wall':
            if 'kind' in kwargs.keys():
                projc, bndry_points = self.wall_bdry_projection2d(kwargs['kind'])
            else:
                projc, bndry_points = self.wall_bdry_projection2d()
        elif bctype == 'circle':
            projc, bndry_points = self.circle_bdry_projection2d()
        elif bctype == 'slit':
            projc, bndry_points = self.slit_bdry_projection2d()
        elif bctype == 'wall_deriv':
            projc, bndry_points = self.wall_deriv_projection2d()
        else:
            raise Exception("Boundary type {bb} not implemented.".format(bb=bctype))

        if not self.eqtype.lower() == 'wave':
            self.bndry_points = bndry_points
            if not 'deriv' in bctype.lower():
                self.projc = PenaltyProj(operator=projc, kind='value')
            else:
                self.projc = PenaltyProj(operator=projc, kind='deriv')
        # If wave equation, need to account for change of variables structure
        # So far also haven't implemented derivative constraints here
        elif self.eqtype.lower() == 'wave':
            assert 'deriv' not in bctype.lower()
            wave_projc = sparse.block_array([[projc, None], [None, projc]])
            wave_bndry_points = wave_projc.diagonal().nonzero()
            self.bndry_points = wave_bndry_points
            # np.array([*zip(*swap.nonzero())],dtype=int)
            self.projc = PenaltyProj(operator=wave_projc, kind='value')
    ''' -------------------------------------------------- '''
    def _set_system_operator(self, coeff):
        ''' set system matrix '''
        if self.eqtype.lower() == 'heat':
            self.operator = heat_operator2d(self.N, dissipativity=coeff, dx=self.dx)
        elif self.eqtype.lower() == 'wave':
            self.operator = wave_operator2d(self.N, speed_of_sound=coeff, dx=self.dx)
        else:
            raise Exception("Equation type {ee} not implemented.".format(bb=eqtype))
    ''' -------------------------------------------------- '''
    def construct_initial_state_centered(self, width, height):
        ''' construct initial state, Gaussian in center of box '''
        assert self.eqtype.lower() == 'heat'
        gaussian = height * np.exp( -1.*(np.square(self.x-1/2)+np.square(self.y-1/2)) / width )
        self.x0 = np.array(gaussian, dtype=complex).flatten()
        # outpoints = circle_points(self.x, self.y, radius=0.4, kind='outside')
        # Make sure that BC is satisfied -- given homogenization, this is always zero for Dirichlet
        if self.projc.kind == 'value':
            self.x0[self.bndry_points] = 0.
        elif self.projc.kind == 'deriv':
            print("Warning: Depending on the projection used the initial condition might have to be adapted.")
            self.x0[self.bndry_points] = 0.
    ''' -------------------------------------------------- '''
    def construct_initial_state_wave(self):
        ''' construct initial state, slit geometry '''
        assert self.eqtype.lower() == 'wave'
        outpoints = circle_points(self.x, self.y, radius=0.1, kind='outside')
        v0 = 0.001*(np.sin(self.x)*np.sin(self.y)).flatten()
        w0 = 0.01 *(np.cos(self.x)*np.cos(self.y)).flatten()
        v0[outpoints] = 0
        w0[outpoints] = 0
        full_x0 = np.zeros((2*self.N**2,), dtype=complex)
        full_x0[:self.N**2] = v0.flatten()
        full_x0[self.N**2:] = w0.flatten()
        self.x0 = full_x0
        # Make sure that BC is satisfied
        self.x0[self.bndry_points] = 0.
    ''' -------------------------------------------------- '''
    # def time_evolution(self,  ode_method=RK45):
    def time_evolution(self,  ode_method=RK23):
        '''
            Function that simulates the time evolution using penalty constraints of the
            system operator defined in the class with respect to the initial conditions x0
            Does not work for the wave equation with the dilation in [u]->[v;w]
            Returns:
            - val_array, which has been corrected by homogenization shift
            - bndry_val_array, to compute error with (not corrected by shift,
                                            should be close to zero everywhere ideally)
        '''
        if self.eqtype.lower() == 'wave':
            val_array, soln_bndry = self._time_evolution_wave(ode_method=ode_method)
            return val_array, soln_bndry
        else:
            ipic = True
            if ipic:
                RHS = lambda t: self.projc.apply_exp_fn(self.lam)(t) @ ( self.RHS + self.L_at_bcvals )
                sol = ode_method(fun=ode_fn(operator=ipic_operator(self.operator, self.projc, self.lam),
                                            RHS=RHS), t0=0, y0=self.x0, t_bound=self.T,
                                            first_step=self.dt,
                                            max_step=self.dt)
            elif not ipic:
                raise Exception('nope')
                '''
                sol = ode_method(fun=ode_fn(lambda t: self.operator - 1.j*self.lam*self.projc.operator, lambda t: self.RHS),
                        t0=0, y0=self.x0, t_bound=self.T, first_step=self.dt, max_step=self.dt)
                '''

            n_steps = int(self.T*(1/self.dt))
            # t_array = np.zeros(n_steps)
            '''
            val_array = np.zeros((n_steps,self.N,self.N), dtype=complex)
            bndry_val_array = np.zeros((n_steps,self.N**2), dtype=complex)
            '''
            # here only store first and last
            val_array = np.zeros((2,self.N,self.N), dtype=complex)
            bndry_val_array = np.zeros((2,self.N**2), dtype=complex)
            val_array[0,:,:] = np.reshape(self.x0, (self.N,self.N))
            bndry_val_array[0,self.bndry_points] =  self.x0[self.bndry_points]

            ''' wanna change this to the scipy.solve_ivp function '''
            for i in range(1, n_steps):
                sol.step()
                # t_array[i] = sol.t
                # convert back from interaction picture
                if ipic:
                    vals = self.projc.apply_exp_fn(self.lam)(-sol.t)@sol.y
                elif not ipic:
                    vals = sol.y
                '''
                val_array[i,:,:] = copy.deepcopy(np.reshape(vals, (self.N,self.N)))
                bndry_val_array[i, self.bndry_points] = copy.deepcopy(vals[self.bndry_points])
                '''
                val_array[-1,:,:] = copy.deepcopy(np.reshape(vals, (self.N,self.N)))
                bndry_val_array[-1, self.bndry_points] = copy.deepcopy(vals[self.bndry_points])
                # if not i%1000:
                #     print('at step', i)

            val_array += self.bcvals.reshape((self.N,self.N))

            print('Finished after ', n_steps, ' steps.')

            return val_array, bndry_val_array
    ''' -------------------------------------------------- '''
    def _time_evolution_wave(self, ode_method=RK45):
        ''' time evolve the wave equation by variable substitution '''
        # wave equation with source term not implemented yet
        ipic = True
        if ipic:
            RHS = lambda t: self.projc.apply_exp_fn(self.lam)(t) @ ( self.RHS + self.L_at_bcvals )
            sol = ode_method(fun=ode_fn(ipic_operator(self.operator, self.projc, self.lam),
                                        RHS), t0=0, y0=self.x0, t_bound=self.T,
                                                first_step=self.dt,
                                                max_step=self.dt)
        elif not ipic:
            raise Exception("This needs to be implemented.")
            RHS = self.RHS + self.L_at_bcvals
            sol = ode_method(fun=ode_fn(lambda t: self.operator - 1.j*self.lam*self.projc, RHS),
                             t0=0, y0=self.x0, t_bound=self.T, first_step=self.dt)

        n_steps = int(self.T*(1/self.dt))
        t_array = np.zeros(n_steps)
        '''
        val_array = np.zeros((n_steps,2,self.N,self.N), dtype=complex)
        soln_bndry = np.zeros((n_steps,2,self.N**2), dtype=complex)
        '''
        val_array = np.zeros((2,2,self.N,self.N), dtype=complex)
        soln_bndry = np.zeros((2,2,self.N**2), dtype=complex)
        val_array[0,0,...] = np.reshape(self.x0[:self.N**2], (self.N,self.N))
        val_array[0,1,...] = np.reshape(self.x0[self.N**2:], (self.N,self.N))
        # we assume that at initial time compliant to bndry conditions so having zero
        # there is fine.
        # I know this deviates from what's implemented in the case above but effectively
        # are just doing the same at the moment.

        for i in range(n_steps):
            sol.step()
            t_array[i] = sol.t
            # convert back from interaction picture
            if ipic:
                vals = self.projc.apply_exp_fn(self.lam)(-sol.t)@sol.y
            elif not ipic:
                raise Exception("This needs to be implemented.")
                # vals = ipic_rotation(self.projc, self.lam)(sol.t)@sol.y
            bvals = np.zeros_like(vals)
            bvals[self.bndry_points] = vals[self.bndry_points]
            '''
            val_array[ i,0,...] = np.reshape(vals[:self.N**2], (self.N,self.N))  # [v]
            val_array[ i,1,...] = np.reshape(vals[self.N**2:], (self.N,self.N))  # [w]
            soln_bndry[i,0,...] = np.reshape(bvals[:self.N**2], (self.N**2))  # [v]
            soln_bndry[i,1,...] = np.reshape(bvals[self.N**2:], (self.N**2))  # [w]
            '''
            val_array[ -1,0,...] = np.reshape(vals[:self.N**2], (self.N,self.N))  # [v]
            val_array[ -1,1,...] = np.reshape(vals[self.N**2:], (self.N,self.N))  # [w]
            soln_bndry[-1,0,...] = np.reshape(bvals[:self.N**2], (self.N**2))  # [v]
            soln_bndry[-1,1,...] = np.reshape(bvals[self.N**2:], (self.N**2))  # [w]
            # if not i%100:
            #     print('at step', i)
        print('Finished after ', n_steps, ' steps.')

        return val_array, soln_bndry
    ''' -------------------------------------------------- '''
    def set_RHS(self, in_vals):
        ''' set source term '''
        self.RHS = in_vals.flatten()
    ''' -------------------------------------------------- '''
    def set_bcvals(self, in_vals):
        ''' set boundary values and associated additional source terms '''
        self.bcvals = in_vals.flatten()
        print('Setting additional source terms according to time-independent BC.')
        self.L_at_bcvals = self.operator @ self.bcvals
