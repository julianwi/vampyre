"""
linear.py:  Linear estimation class
"""
from __future__ import division

import numpy as np
import scipy.sparse.linalg

# Import other subpackages in vampyre
import vampyre.common as common
import vampyre.trans as trans

# Import individual classes and methods from the current sub-package
from vampyre.estim.base import BaseEst


class LinEst(BaseEst):
    """
    Estimator based on a linear constraint with noise

    This estimator corresponds to a linear constraint of the form
    :math:`y = Az + w`
    where :math:`w \\sim {\\mathcal N}(0,\\tau_w I)`.
    Specifically, the penalty is

    :math:`f(z) = (1/2\\tau_w)\|y-Az\|^2 + (d/2)\\ln(2\\pi \\tau_w)`,

    where :math:`d` is the dimension of :math:`y`

    :param A: Linear operator represented as a
        :class:`vampyre.trans.base.LinTrans`
    :param y:  Output
    :param wvar:  Noise level
    :param wrep_axes':  The axes on which the output noise variance is repeated.
        Default is 'all'.
    :param var_axes:  The axes on which the input variance is averaged.
    :param tune_wvar:  Enables tuning of noise level.  In this case,
        :code:`wvar` is used as an initial condition.
    :param Boolean is_complex:  indiates if :math:`z` is complex
    :param Boolean map_est:  indicates if estimator is to perform MAP
        or MMSE estimation. This is used for the cost computation.
    :param rvar_init:  Initial prior variance used in the
        :code:`est_init` method.

    :note:  The linear operator :code:`A` must have :code:`svd_avail==True`.
       In the future, if an SVD is not available, we will use conjugate
       gradient
    :note:  The linear operator must also have the :code:`shape0` and
       :code:`shape1` arrays available to compute the dimensions.

    :note:  The axes :code:`wrep_axes` and :code:`zerp_axes` must
       include the axis in which :code:`A` operates.
    """
    def __init__(self,A,y,wvar=0,\
                 wrep_axes='all', var_axes=(0,),name=None,map_est=False,\
                 is_complex=False,rvar_init=1e5,tune_wvar=False, est_meth='svd', nit_cg=100, atol_cg=1e-10):

        BaseEst.__init__(self, shape=A.shape0, var_axes=var_axes,\
            dtype=A.dtype0, name=name,\
            type_name='LinEstim', nvars=1, cost_avail=True)
        self.A = A
        self.y = y
        self.wvar = wvar
        self.map_est = map_est
        self.is_complex = is_complex
        self.cost_avail = True
        self.rvar_init = rvar_init
        self.tune_wvar = tune_wvar

        # Get the input and output shape
        self.zshape = A.shape0
        self.yshape = A.shape1

        # Set the repetition axes
        ndim = len(self.yshape)
        if wrep_axes == 'all':
            wrep_axes = tuple(range(ndim))
        self.wrep_axes = wrep_axes

        # Initialization depending on the estimation method
        self.est_meth = est_meth
        self.nit_cg = nit_cg
        self.atol_cg = atol_cg
        if self.est_meth == 'svd':
            self.init_svd()
        elif self.est_meth == 'cg':
            self.init_cg()
        else:
            raise common.VpException(
                "Unknown estimation method {0:s}".format(est_meth))

    def init_cg(self):
        """
        Initialization that is specific to the conjugate gradient method
        """

        # Draw random perturbations for computing the numerical gradients
        grad_step = .01
        self.dr0 = np.random.normal(0,grad_step,self.zshape)
        self.dr0_norm_sq = np.mean(np.abs(self.dr0)**2, self.var_axes)

        # Initialize the variables
        self.zlast = None
        self.zvec0_last = None

    def init_svd(self):
        """
        Initialization for the SVD method 
        """

        # Compute the SVD terms
        # Take an SVD A=USV'.  Then write p = SV'z + w,
        if not self.A.svd_avail:
            raise common.VpException("Transform must support an SVD")
        self.p = self.A.UsvdH(self.y)
        srep_axes = self.A.get_svd_diag()[2]

        # Compute the norm of ||y-UU*(y)||^2/wvar
        if np.all(self.wvar > 0):
            yp = self.A.Usvd(self.p)
            wvar1 = common.repeat_axes(self.wvar, self.yshape, self.wrep_axes, rep=False)
            err = np.abs(self.y-yp)**2
            self.ypnorm = np.sum(err/wvar1)
        else:
            self.ypnorm = 0

        # Check that all axes on which A operates are repeated
        ndim = len(self.yshape)
        for i in range(ndim):
            if not (i in self.wrep_axes) and not (i in srep_axes):
                raise common.VpException(
                    "Variance must be constant over output axis")
            if not (i in self.var_axes) and not (i in srep_axes):
                raise common.VpException(
                    "Variance must be constant over input axis")


    def est_init(self, return_cost=False, ind_out=None,\
        avg_var_cost=True):
        """
        Initial estimator.

        See the base class :class:`vampyre.estim.base.Estim` for
        a complete description.

        :param boolean return_cost:  Flag indicating if :code:`cost` is
            to be returned
        :returns: :code:`zmean, zvar, [cost]` which are the
            prior mean and variance
        """
        # Check parameters
        if (ind_out != [0]) and (ind_out != None):
            raise ValueError("ind_out must be either [0] or None")
        if not avg_var_cost:
            raise ValueError("disabling variance averaging not supported for LinEst")

        if self.est_meth == 'cg':
            return self.est_cg(np.zeros(self.zshape), np.mean(np.full(self.zshape,np.Inf), axis=self.var_axes), return_cost=return_cost)

        # Get the diagonal parameters
        s, sshape, srep_axes = self.A.get_svd_diag()
        shape0 = self.A.shape0

        # Reshape the variances to the transformed space
        s1    = common.repeat_axes(s, sshape, srep_axes)
        wvar1 = common.repeat_axes(self.wvar, sshape, self.wrep_axes, rep=False)

        # Compute the estimate within the transformed space
        q = (1/s1)*self.p
        qvar = wvar1/(np.abs(s1)**2)
        qvar_mean = np.mean(qvar, axis=self.var_axes)

        rdim = np.product(sshape)/np.product(shape0)
        zmean = self.A.Vsvd(q)
        zvar = rdim*qvar_mean + (1-rdim)*self.rvar_init

        # Exit if cost does not need to be computed
        if not return_cost:
            return zmean, zvar

        # Computes the MAP output cost
        if np.all(self.wvar > 0):
            cost = self.ypnorm
        else:
            cost = 0

        # Compute the output variance cost
        if np.all(self.wvar > 0) and self.map_est:
            clog = np.log(2*np.pi*self.wvar)
            cost += common.repeat_sum(clog, self.zshape, self.wrep_axes)

        # Scale for real case
        if not self.is_complex:
            cost = 0.5*cost
        return zmean, zvar, cost


    def est(self,r,rvar,return_cost=False, ind_out=None,\
        avg_var_cost=True):
        """
        Estimation function

        The proximal estimation function as
        described in the base class :class:`vampyre.estim.base.Estim`

        :param r: Proximal mean
        :param rvar: Proximal variance
        :param Boolean return_cost:  Flag indicating if :code:`cost` is
            to be returned

        :returns: :code:`zhat, zhatvar, [cost]` which are the posterior
            mean, variance and optional cost.
        """

        # Check parameters
        if (ind_out != [0]) and (ind_out != None):
            raise ValueError("ind_out must be either [0] or None")
        if not avg_var_cost:
            raise ValueError("disabling variance averaging not supported for LinEst")

        if self.est_meth == 'svd':
            return self.est_svd(r,rvar,return_cost)
        elif self.est_meth == 'cg':
            return self.est_cg(r,rvar,return_cost)
        else:
            raise common.VpException(
                "Unknown estimation method {0:s}".format(self.est_meth))

    def est_svd(self,r,rvar,return_cost=False):
        # Get the diagonal parameters
        s, sshape, srep_axes = self.A.get_svd_diag()

        # Get dimensions
        nz = np.prod(self.zshape)
        ny = np.prod(self.yshape)
        ns = np.prod(sshape)

        # Reshape the variances to the transformed space
        s1    = common.repeat_axes(s, sshape, srep_axes, rep=False)
        rvar1 = common.repeat_axes(rvar, sshape, self.var_axes, rep=False)
        wvar1 = common.repeat_axes(self.wvar, sshape, self.wrep_axes, rep=False)

        # Compute the estimate within the transformed space
        qbar = self.A.VsvdH(r)
        d = 1/(rvar1*(np.abs(s1)**2) + wvar1)
        q = d*(rvar1*s1.conj()*self.p + wvar1*qbar)
        qvar = rvar1*wvar1*d
        qvar_mean = np.mean(qvar, axis=self.var_axes)

        zhat = self.A.Vsvd(q - qbar) + r
        zhatvar = ns/nz*qvar_mean + (1-ns/nz)*rvar

        # Update the variance estimate if tuning is enabled
        if self.tune_wvar:
            yerr = np.abs(self.y - self.A.Usvd(s1*q))**2
            self.wvar = np.mean(yerr, self.wrep_axes) + np.mean(qvar*(np.abs(s1)**2),self.wrep_axes)

        # Exit if cost does not need to be computed
        if not return_cost:
            return zhat, zhatvar

        # Computes the MAP output cost
        if np.all(self.wvar > 0):
            err = np.abs(self.p-s1*q)**2
            cost = self.ypnorm + np.sum(err/wvar1)
        else:
            cost = 0

        # Add the MAP input cost
        err = np.abs(q-qbar)**2
        cost = cost + np.sum(err/rvar1)

        # Compute the variance cost.
        if self.map_est:
            # For the MAP case, this is log(2*pi*wvar)
            if np.all(self.wvar > 0):
                cost += ny*np.mean(np.log(2*np.pi*self.wvar))
        else:
            # For the MMSE case, this is 1 + log(2*pi*wvar) - H(b)
            # where b is the Gaussian with variance wvar*rvar*d
            cost +=  -ns*np.mean(np.log(rvar1*d)) -\
                (nz-ns)*np.mean(np.log(2*np.pi*rvar1))
            if np.all(self.wvar > 0):
                cost += (ny-ns)*np.mean(np.log(2*np.pi*self.wvar))

        # Scale for real case
        if not self.is_complex:
            cost = 0.5*cost

        return zhat, zhatvar, cost

    def est_cg(self,r,rvar,return_cost=False):
        """
        First-order terms
        """
        # Create the LSQR transform for the problem
        # The VAMP problem is equivalent to minimizing ||F(z)-g||^2
        F = LSQROp(self.A,self.y,rvar, self.wvar,\
            self.var_axes, self.wrep_axes,\
            self.zshape, self.yshape, self.is_complex)
        g = F.get_tgt_vec(r)

        # Get the initial condition
        if self.zlast is None:
            zinit = F.pack(r)
        else:
            zinit = self.zlast
        g -= F.dot(zinit)

        # Run the LSQR optimization
        lsqr_out = scipy.sparse.linalg.lsqr(F,g,iter_lim=self.nit_cg,atol=self.atol_cg)
        zvec = lsqr_out[0] + zinit
        self.zlast = zvec
        zhat = F.unpack(zvec)

        """
        Cost
        """
        if return_cost:
            # Compute the cost
            cost = lsqr_out[3]**2
    
            # Add the cost for the second order terms.
            # 
            # We only consider the MAP case, where the second-order cost is
            # (1/2)*ny*log(2*pi*wvar)
            if self.is_complex:
                cscale = 1
            else:
                cscale = 2
            cost /= cscale
            if np.all(self.wvar > 0):
                ny = np.prod(self.yshape)
                cost += (1/cscale)*ny*np.mean(np.log(cscale*np.pi*self.wvar))

        """
        Second-order terms

        These are computed via the numerical gradient along a random direction
        """
        if np.any(rvar==np.Inf):
            # Create the LSQR transform for the problem
            # The VAMP problem is equivalent to minimizing ||F(z)-g||^2
            F = LSQROp(self.A,np.random.normal(self.y,np.sqrt(self.wvar),self.y.shape),rvar, self.wvar,\
                self.var_axes, self.wrep_axes,\
                self.zshape, self.yshape, self.is_complex)
            g = F.get_tgt_vec(r)

            # Get the initial condition
            zinit = F.pack(r)
            g -= F.dot(zinit)

            # Run the LSQR optimization
            lsqr_out = scipy.sparse.linalg.lsqr(F,g,iter_lim=self.nit_cg,atol=self.atol_cg)
            zvec = lsqr_out[0] + zinit
            zhat_ = F.unpack(zvec)
            zhatvar = np.mean(np.abs(zhat_-zhat)**2, axis=self.var_axes)

            sshape = np.array(self.zshape)
            sshape[-1-(len(self.zshape)-1)] = min(self.zshape[-1-(len(self.zshape)-1)], self.yshape[-1-(len(self.yshape)-1)])
            sshape = tuple(sshape)
            shape0 = self.A.shape0
            rdim = np.product(sshape)/np.product(shape0)
            zhatvar = rdim*zhatvar + (1-rdim)*self.rvar_init

        else:
            # Perturb r0
            r0p = r + self.dr0
            g0 = F.get_tgt_vec(r0p)

            # Get the initial condition
            if self.zvec0_last is None:
                zinit = F.pack(r0p)
            else:
                zinit = self.zvec0_last
            g0 -= F.dot(zinit)

            # Run the LSQR optimization
            lsqr_out = scipy.sparse.linalg.lsqr(F,g0,iter_lim=self.nit_cg,atol=self.atol_cg)
            zvec0 = lsqr_out[0] + zinit
            self.zvec0_last = zvec0
            dzvec = zvec0 - zvec
            dz0 = F.unpack(dzvec)

            # Compute the correlations
            alpha0 = np.mean(np.real(self.dr0.conj()*dz0),self.var_axes) /\
                self.dr0_norm_sq
            zhatvar = alpha0*rvar

        if return_cost:
            return zhat,zhatvar, cost
        else:
            return zhat,zhatvar

class LSQROp(scipy.sparse.linalg.LinearOperator):
    """
    LSQR operator for the VAMP least squares problem.

    Defines an operator F(z) and constant vector g such that the VAMP
    optimization is equivalent to

    min_z ||F(z) - g||^2

    This can be solved with LSQR.

        F(z) = D*[z; A.dot(z)]  g = D*[r; b]
        D = diag(1/sqrt([rvar; wvar]))
    """
    def __init__(self,A,b,rvar,wvar,zrep_axes,wrep_axes,\
        shape0,shape1,is_complex):
        self.A = A
        self.b = b
        self.shape0 = shape0
        self.shape1 = shape1
        self.n0 = np.prod(shape0)
        self.n1 = np.prod(shape1)
        
        # Compute scale factors
        self.rsqrt = common.repeat_axes(
            np.sqrt(rvar), self.shape0, zrep_axes, rep=False)        
        self.wsqrt = common.repeat_axes(
            np.sqrt(wvar), self.shape1, wrep_axes, rep=False)

        # Compute dimensions of the transform F
        nin = self.n0
        nout = self.n0 + self.n1
        self.shape = (nout,nin)
        if is_complex:
            self.dtype = np.dtype(complex)
        else:
            self.dtype = np.dtype(float)

    def unpack(self,zvec):
        """
        Unpacks the variables from vector for the CG estimation
        """
        z = zvec.reshape(self.shape0)
        return z

    def pack(self,z):
        """
        Packs the variables from vector for the CG estimation to a vector
        """
        zvec = z.ravel()
        return zvec
    
    def get_tgt_vec(self,r):
        """
        Computes the target vector `g` in the above description
        """
        g0 = 1/self.rsqrt*r
        g1 = 1/self.wsqrt*(self.b)
        g = np.hstack((g0.ravel(),g1.ravel()))
        return g

    def _matvec(self,r):
        """
        Forward multiplication for the operator `F` defined above
        """
        r0 = self.unpack(r)
        y0 = 1/self.rsqrt*r0
        yout = 1/self.wsqrt*self.A.dot(r0)
        y = np.hstack((y0.ravel(), yout.ravel()))
        return y

    def _rmatvec(self,y):
        """
        Adjoint multiplication for the operator `F` defined above
        """
        y0 = np.reshape(y[:self.n0], self.shape0)
        y1 = np.reshape(y[self.n0:], self.shape1)
        r0 = 1/self.rsqrt*y0
        r0 += 1/self.wsqrt*self.A.dotH(y1)
        r = r0.ravel()
        return r
