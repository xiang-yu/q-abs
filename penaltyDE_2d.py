from penalty_utils import *
import copy

from scipy.fftpack import dct, idct, dst, idst


class PenaltyProj:
    '''
        convenience class for projection operator to provide time evolution
    '''
    def __init__(self, operator, kind):
        self.kind = kind.lower()
        if self.kind not in {'value', 'deriv'}:
            raise ValueError("kind must be either 'value' or 'deriv'")

        self.operator = sparse.csc_matrix(operator, dtype=complex)
        if self.operator.shape[0] != self.operator.shape[1]:
            raise ValueError('projection operator must be square')
        ident = sparse.identity(
            self.operator.shape[0], dtype=complex, format='csc'
        )
        self.complement = (ident - self.operator).tocsc()

    def apply_exp_fn(self, lam):
        '''Return ``exp(i * lam * t * P)`` using projector fast-forwarding.

        Since ``P`` is an orthogonal projector, its exponential is exactly
        ``(I - P) + exp(i * lam * t) P``.  Its construction therefore costs
        only O(nnz(P)) and is independent of the magnitude of ``lam * t``.
        '''
        return lambda t: (
            self.complement + np.exp(1.j * lam * t) * self.operator
        )

    def apply_exp_to(self, values, lam, t):
        '''Apply ``exp(i * lam * t * P)`` without constructing a matrix.'''
        values = np.asarray(values)
        phase = np.exp(1.j * lam * t)
        return values + (phase - 1.) * (self.operator @ values)

    def project_feasible(self, values):
        '''Project values onto the constraint-satisfying kernel of ``P``.'''
        return self.complement @ np.asarray(values)

    def constraint_residual(self, values):
        '''Return the component that violates the encoded constraint.'''
        return self.operator @ np.asarray(values)
""" ========================================== """
class PenaltyDE_2d:
    '''
        convenience class to implement penalty projection approach
        using interaction picture simulation for 2d discretized PDEs
    '''
    def __init__(self, N=int(2**5), dt=1e-2, T=1, lam=1e6, eqtype='heat', coeff=(1,1), bctype='wall', **kwargs):
        self.N = N
        self.dt = dt
        self.T = T
        self.lam = lam
        self.x, self.y, self.dx = self._create_xy()
        self.eqtype = eqtype.lower()
        self.bctype = bctype.lower()
        self.coeff = coeff
        self._set_system_operator(coeff=coeff)

        self.x0 = None
        self.bndry_points = None # , bndry_vals = None, None

        self.projc = None
        self.spatial_projc = None
        self.spatial_bndry_points = None
        self._set_projector(self.bctype, **kwargs)
        if not self.eqtype == 'wave':
            self.RHS         = np.zeros((self.N*self.N), dtype=complex)
            self.bcvals      = np.zeros((self.N*self.N), dtype=complex)
            self.L_at_bcvals = np.zeros((self.N*self.N), dtype=complex)
        elif self.eqtype == 'wave':
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
    def wall_deriv_projection2d(self, kind='lrtb'):
        '''Construct the zero-Neumann projector for selected box walls.'''
        proj = wall_deriv_projection2d(
            self.N, self.x, self.y, kind=kind
        )
        return proj, proj.diagonal().nonzero()
    ''' -------------------------------------------------- '''
    def _set_projector(self, bctype, **kwargs):
        '''
            set projection operation depending on boundary condition
            todo for future: mixed conditions with mixed/non-overlapping projections
        '''
        bctype = bctype.lower()
        proj_kind = 'value'
        if bctype == 'wall':
            if 'kind' in kwargs.keys():
                projc, bndry_points = self.wall_bdry_projection2d(kwargs['kind'])
            else:
                projc, bndry_points = self.wall_bdry_projection2d()
        elif bctype == 'circle':
            projc, bndry_points = self.circle_bdry_projection2d()
        elif bctype == 'slit':
            projc, bndry_points = self.slit_bdry_projection2d()
        elif bctype in {'neumann', 'wall_neumann', 'wall_deriv'}:
            projc, bndry_points = self.wall_deriv_projection2d(
                kwargs.get('kind', 'lrtb')
            )
            proj_kind = 'deriv'
        else:
            raise ValueError("Boundary type {bb} not implemented.".format(bb=bctype))

        self.spatial_projc = sparse.csc_matrix(projc, dtype=complex)
        self.spatial_bndry_points = bndry_points
        if self.eqtype != 'wave':
            self.bndry_points = self.spatial_bndry_points
            self.projc = PenaltyProj(
                operator=self.spatial_projc, kind=proj_kind
            )
        else:
            # If d_n u = 0 for all t, then d_n w = d_t(d_n u) = 0 as
            # well.  Apply the same spatial constraint to u and w.
            wave_projc = sparse.block_diag(
                (self.spatial_projc, self.spatial_projc), format='csc'
            )
            self.bndry_points = wave_projc.diagonal().nonzero()
            self.projc = PenaltyProj(
                operator=wave_projc, kind=proj_kind
            )
    ''' -------------------------------------------------- '''
    def _set_system_operator(self, coeff):
        ''' set system matrix '''
        if self.eqtype.lower() == 'heat':
            self.operator = heat_operator2d(self.N, dissipativity=coeff, dx=self.dx)
        elif self.eqtype.lower() == 'wave':
            self.operator = wave_operator2d(self.N, speed_of_sound=coeff, dx=self.dx)
        else:
            raise ValueError("Equation type {ee} not implemented.".format(ee=self.eqtype))
    ''' -------------------------------------------------- '''
    def construct_initial_state_centered(self, width, height):
        ''' construct initial state, Gaussian in center of box '''
        assert self.eqtype.lower() == 'heat'
        gaussian = height * np.exp( -1.*(np.square(self.x-1/2)+np.square(self.y-1/2)) / width )
        self.x0 = np.array(gaussian, dtype=complex).flatten()
        # outpoints = circle_points(self.x, self.y, radius=0.4, kind='outside')
        # Make sure the homogeneous value or derivative constraint is met.
        self.x0 = self.projc.project_feasible(self.x0)
    ''' -------------------------------------------------- '''
    def construct_initial_state_wave(self):
        '''Construct the compactly supported initial wave state.'''
        assert self.eqtype.lower() == 'wave'
        outpoints = circle_points(self.x, self.y, radius=0.1, kind='outside')
        v0 = 0.001*(np.sin(self.x)*np.sin(self.y)).flatten()
        w0 = 0.01 *(np.cos(self.x)*np.cos(self.y)).flatten()
        v0[outpoints] = 0
        w0[outpoints] = 0
        full_x0 = np.zeros((2*self.N**2,), dtype=complex)
        full_x0[:self.N**2] = v0.flatten()
        full_x0[self.N**2:] = w0.flatten()
        # Make sure that the value or derivative constraint is satisfied for
        # both u and w without unnecessarily zeroing Neumann boundary pairs.
        self.x0 = self.projc.project_feasible(full_x0)
    ''' -------------------------------------------------- '''
    # def time_evolution(self,  ode_method=RK45):
    def time_evolution(self,  ode_method=RK23):
        '''
            Function that simulates the time evolution using penalty constraints of the
            system operator defined in the class with respect to the initial conditions x0
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
                rhs_vals = self.RHS + self.L_at_bcvals
                RHS = lambda t: self.projc.apply_exp_to(
                    rhs_vals, self.lam, t
                )
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
            bndry_val_array[0,:] = self.projc.constraint_residual(self.x0)
            ''' wanna change this to the scipy.solve_ivp function '''
            for i in range(1, n_steps):
                sol.step()
                # t_array[i] = sol.t
                # convert back from interaction picture
                if ipic:
                    vals = self.projc.apply_exp_to(
                        sol.y, self.lam, -sol.t
                    )
                elif not ipic:
                    vals = sol.y
                '''
                val_array[i,:,:] = copy.deepcopy(np.reshape(vals, (self.N,self.N)))
                bndry_val_array[i, self.bndry_points] = copy.deepcopy(vals[self.bndry_points])
                '''
                val_array[-1,:,:] = copy.deepcopy(np.reshape(vals, (self.N,self.N)))
                bndry_val_array[-1,:] = copy.deepcopy(
                    self.projc.constraint_residual(vals)
                )
                # if not i%1000:
                #     print('at step', i)
            val_array += self.bcvals.reshape((self.N,self.N))

            print('Finished after ', n_steps, ' steps.')
            return val_array, bndry_val_array
    ''' -------------------------------------------------- '''
    def _time_evolution_wave(self, ode_method=RK45):
        ''' time evolve the wave equation by variable substitution '''
        ipic = True
        if ipic:
            rhs_vals = self.RHS + self.L_at_bcvals
            RHS = lambda t: self.projc.apply_exp_to(
                rhs_vals, self.lam, t
            )
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
        initial_residual = self.projc.constraint_residual(self.x0)
        soln_bndry[0,0,...] = initial_residual[:self.N**2]
        soln_bndry[0,1,...] = initial_residual[self.N**2:]
        for i in range(n_steps):
            sol.step()
            t_array[i] = sol.t
            # convert back from interaction picture
            if ipic:
                vals = self.projc.apply_exp_to(
                    sol.y, self.lam, -sol.t
                )
            elif not ipic:
                raise Exception("This needs to be implemented.")
                # vals = ipic_rotation(self.projc, self.lam)(sol.t)@sol.y
            bvals = self.projc.constraint_residual(vals)
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
        new_bcvals = in_vals.flatten()
        if self.projc.kind == 'deriv' and np.any(new_bcvals):
            raise NotImplementedError(
                'Only homogeneous (zero) Neumann data is implemented.'
            )
        self.bcvals = new_bcvals
        print('Setting additional source terms according to time-independent BC.')
        self.L_at_bcvals = self.operator @ self.bcvals
