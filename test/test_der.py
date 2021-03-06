'''
@author: salvatore maraniello
@date: 12 Jul 2017
@brief: test computation of linearised term for linear solution

'''
import os, sys
sys.path.append('..')

import numpy  as np
import lin_uvlm2d_sta
import lin_uvlm2d_dyn
import uvlm2d_sta


import warnings
import unittest
from IPython import embed
import matplotlib.pyplot as plt



def max_error_tensor(Pder_an,Pder_num):
	'''
	Finds the maximum error analytical derivatives Pder_an. The error is:
	- relative, if the element of Pder_an is nonzero
	- absolute, otherwise

	The function returns the absolute and relative error tensors, and the
	maximum error. 

	@warning: The relative error tensor may contain NaN or Inf if the
	analytical derivative is zero. These elements are filtered out during the
	search for maximum error, and absolute error is checked.
	'''

	Eabs=Pder_num-Pder_an

	Erel=Eabs/Pder_an

	# Relative error check: remove NaN and inf...
	iifinite=np.isfinite(Erel)
	err_max=0.0
	for err_here in Erel[iifinite]:
		if np.abs(err_here)>err_max:
			err_max=err_here

	# Zero elements check
	iinonzero=np.abs(Pder_an)<1e-15
	for der_here in Pder_num[iinonzero]:
		if np.abs(der_here)>err_max:
			err_max=der_here 

	return err_max, Eabs, Erel



class Test_der_sta(unittest.TestCase):
	'''
	Each method defined in this class contains a test case.
	@warning: by default, only functions whose name starts with 'test' is run.
	'''


	def setUp(self):
		''' Common piece of code to initialise the test '''

		# build reference state
		Mw=4
		M=3
		alpha=0.*np.pi/180.
		ainf=15.0*np.pi/180.
		Uinf=20.*np.array([np.cos(ainf),np.sin(ainf)])
		chord=5.0
		b=0.5*chord
		self.S0=uvlm2d_sta.solver(M,Mw,b,Uinf,alpha,rho=1.225)
		#self.S0.build_flat_plate()
		self.S0.build_camber_plate(Mcamb=50,Pcamb=4)
		self.S0.solve_static_Gamma2d()

		# linear solver
		self.Slin=lin_uvlm2d_sta.solver(self.S0)


	def test_der_WncV_dZeta(self):
		'''
		Test lin_uvlm2d_sta.solver.der_WncV_dZeta against FD

		@note: for this derivative, it is not important to normalise variables
		'''

		S0,Slin=self.S0,self.Slin
		M,K=S0.M,S0.K
		Ndim=2

		# Compute WncV:
		Pder_an=Slin.der_WncV0_dZeta(S0.Vcoll)


		### compute numerically:

		# reference values
		Zeta=S0.Zeta.copy()
		Vcollperp=np.dot(S0._Wnc[0,:,:],S0.Vcoll[:,0])+\
		                                    np.dot(S0._Wnc[1,:,:],S0.Vcoll[:,1])

		FDstep=1e-6*S0.b
		Pder_num=np.zeros((2,M,K))

		for kk in range(K):
			for dd in range(Ndim):

				# reset grid
				S0.Zeta=Zeta.copy()
				# Perturb grid
				S0.Zeta[kk,dd]+=FDstep
				# update normal and convertion matrices
				S0.get_Nmat()
				S0.get_Wmats()
				# produce V at collocation points (should not change)
				Vcoll=np.dot(S0._Wcv,S0.Uzeta+S0.Wzeta-S0.dZetadt)
				# compute normal component
				Vcollperp_here=np.dot(S0._Wnc[0,:,:],Vcoll[:,0])+\
		                                       np.dot(S0._Wnc[1,:,:],Vcoll[:,1])
		        # get derivative
				Pder_num[dd,:,kk]=(Vcollperp_here-Vcollperp)/FDstep

		# check maximum error
		err_max,Eabs,Erel=max_error_tensor(Pder_an,Pder_num)
		tol=10.*FDstep
		assert err_max<tol,\
			      'Maximum error %.2e largern than tolerance %.2e'%(err_max,tol)
		#embed()



	def test_der_Wnc0AGamma_dZeta(self):
		'''
		Test lin_uvlm2d_sta.solver.der_Wnc0AGamma_dZeta against FD. In order
		to compute this term only, the normal vectors are not updated during
		the FD step.

		@warning: variables should be normalised!
		'''

		S0,Slin=self.S0,self.Slin
		M,Mw,K=S0.M,S0.Mw,S0.K
		Ndim=2

		S0.nondimvars()

		# regularise circulation to have constant jumps along segments. This
		# avoids some terms vanishing when the net circulation over a segment
		# approaches zero
		S0.Gamma=np.linspace(3,3*S0.M,S0.M)
		S0.GammaW=np.linspace(2,2*S0.Mw,S0.Mw)

		# Compute Induced velocity (3 comp) at collocation points
		#Vind_bound=np.zeros((M,Ndim))
		#Vind_bound[:,0]=np.dot(S0.AA[0,:,:],S0.Gamma)
		#Vind_bound[:,1]=np.dot(S0.AA[1,:,:],S0.Gamma)
		Vind_bound=np.dot(S0.AA,S0.Gamma).T 
		Vind_bound_wake=np.dot(S0.AAWW,S0.GammaW).T 

		# Analytical derivative
		Pder_an, PderW_an=Slin.der_Wnc0AGamma_dZeta()
		Pder_an=Pder_an+Slin.der_WncV0_dZeta(Vind_bound)
		PderW_an=PderW_an+Slin.der_WncV0_dZeta(Vind_bound_wake)

		### Numerical derivatives:
		# reference values
		Zeta=S0.Zeta.copy()
		Zeta_c=S0.Zeta_c.copy()
		Vind_perp=np.dot(S0.A,S0.Gamma)

		ZetaW=S0.ZetaW.copy()
		Vind_perp_wake=np.dot(S0.AW,S0.GammaW)

		Aref=S0.A.copy()
		FDstep=1e-6*S0.b
		Pder_num=np.zeros((2,M,K))
		PderW_num=np.zeros((2,M,K))

		for kk in range(K):
			for dd in range(Ndim):

				# reset grid
				S0.Zeta=Zeta.copy()
				S0.ZetaW=ZetaW.copy()

				# Perturb grid
				S0.Zeta[kk,dd]+=FDstep
				if kk==S0.K-1:
					S0.ZetaW[0,dd]+=FDstep
				# update normal and convertion matrices
				S0.get_Nmat()
				S0.get_Wmats()
				# update collocation points
				S0.Zeta_c=np.dot(S0._Wcv,S0.Zeta)  
				# compute induced velocity
				S0.build_AIC_Gamma2d()
				Vind_perp_here=np.dot(S0.A,S0.Gamma)
				Vind_perp_wake_here=np.dot(S0.AW,S0.GammaW)

		        # get derivative
				Pder_num[dd,:,kk]=(Vind_perp_here-Vind_perp)/FDstep
				PderW_num[dd,:,kk]=(Vind_perp_wake_here-Vind_perp_wake)/FDstep

		# check maximum error
		err_max,Eabs,Erel=max_error_tensor(Pder_an,Pder_num)
		errW_max,EWabs,EWrel=max_error_tensor(PderW_an,PderW_num)

		tol=10.*FDstep # increased tolerance: max. error from almost zero term
		assert err_max<tol,\
			       'Maximum error %.2e larger than tolerance %.2e'%(err_max,tol)
		assert errW_max<tol,\
			 'Wake maximum error %.2e larger than tolerance %.2e'%(errW_max,tol)

		#embed()


	def test_der_Fjouk_dZeta_vind_all(self):
		'''
		Test lin_uvlm2d_sta.solver.der_Fjouk_dZeta_vind AND
		lin_uvlm2d_sta.der_Fjouk_dZeta_vind_by_gamma against FD.

		@warning: variables should be normalised!
		'''

		S0,Slin=self.Slin.S0,self.Slin
		M,Mw,K=S0.M,S0.Mw,S0.K
		Ndim=2
		S0.nondimvars()

		# redefine circulation
		S0.Gamma=np.array([1.2*(ii-0.7)*(ii+1) for ii in range(M)])
		S0.GammaW=np.array([2.3*(ii-0.3)*(ii+1.5) for ii in range(Mw)])
		

		S0.gamma=np.dot(S0._TgG,S0.Gamma)
		S0.gammaW=np.dot(S0._TgG_w,S0.GammaW)+np.dot(S0._EgG,S0.Gamma)

		# update induced/total velocity
		S0.get_induced_velocity()
		Vind_zeta=S0.Vind_zeta.copy()
		Vtot_zeta=S0.Uzeta+S0.Wzeta-S0.dZetadt+S0.Vind_zeta

		# Analytical derivatives
		Pder_an=Slin.der_Fjouk_dZeta_vind()
		PderG_an=Slin.der_Fjouk_dZeta_vind_by_gamma()
		Iseg,Icoll=Slin.S0.get_force_matrices()

		# Force Joukovski at segments
		for nn in range(M):
			S0.Faero[nn,:]=-S0.gamma[nn]*\
			                        np.array([-Vtot_zeta[nn,1],Vtot_zeta[nn,0]])
		# Force Joukovski at grid
		Fgrid=S0.Faero.copy()

		# store reference value
		gamma=S0.gamma.copy()
		Gamma=S0.Gamma.copy()
		Zeta=S0.Zeta.copy()
		ZetaW=S0.ZetaW.copy()

		# Go FDs!
		FDstep=1e-6
		Pder_num=np.zeros((Ndim,K,Ndim,K))

		for kk in range(K):
			for dd in range(Ndim):

				# reset grid
				S0.Zeta=Zeta.copy()
				S0.ZetaW=ZetaW.copy()

				# Perturb grid
				S0.Zeta[kk,dd]+=FDstep
				if kk==S0.K-1:
					S0.ZetaW[0,dd]+=FDstep

				# compute induced velocity
				S0.get_induced_velocity()
				S0.Vtot_zeta=S0.Uzeta+S0.Wzeta-S0.dZetadt+S0.Vind_zeta

				# Force Joukovski at segments
				for nn in range(M):
					S0.Faero[nn,:]=-S0.gamma[nn]*\
					          np.array([-S0.Vtot_zeta[nn,1],S0.Vtot_zeta[nn,0]])
				# Force Joukovski at grid
				Fgrid_here=S0.Faero

				# get derivative
				for ii in range(Ndim):
					Pder_num[ii,:,dd,kk]=(Fgrid_here[:,ii]-Fgrid[:,ii])/FDstep

		# check maximum error
		err_max,Eabs,Erel=max_error_tensor(Pder_an,Pder_num)
		errG_max,EabsG,ErelG=max_error_tensor(PderG_an,Pder_num)

		tol=50.*FDstep # increased tolerance: max. error from almost zero term
		assert err_max<tol,\
			       'Maximum error %.2e larger than tolerance %.2e'%(err_max,tol)
		assert errG_max<tol,\
			 'Wake maximum error %.2e larger than tolerance %.2e'%(errW_max,tol)

		#embed()


	def test_der_Fjouk_dV(self):
		'''
		Test lin_uvlm2d_sta.solver.der_Fjouk_dV against FD.

		@warning: variables should be normalised!
		'''

		S0,Slin=self.S0,self.Slin
		M,Mw,K=S0.M,S0.Mw,S0.K
		Ndim=2
		S0.nondimvars()

		# redefine circulation
		S0.Gamma=np.array([1.2*(ii-0.7)*(ii+1) for ii in range(M)])
		S0.GammaW=np.array([2.3*(ii-0.3)*(ii+1.5) for ii in range(Mw)])
		S0.gamma=np.dot(S0._TgG,S0.Gamma)
		S0.gammaW=np.dot(S0._TgG_w,S0.GammaW)+np.dot(S0._EgG,S0.Gamma)

		# Analytical derivative
		Pder_an=Slin.der_Fjouk_dV()
		Iseg,Icoll=Slin.S0.get_force_matrices()

		### Numerical derivatives:
		# reference values
		S0.get_induced_velocity()
		S0.Vtot_zeta=S0.Uzeta+S0.Wzeta-S0.dZetadt+S0.Vind_zeta
		# Force Joukovski at segments
		for nn in range(M):
			S0.Faero[nn,:]=-S0.gamma[nn]*\
			          np.array([-S0.Vtot_zeta[nn,1],S0.Vtot_zeta[nn,0]])
		# Force Joukovski at grid
		Fgrid=S0.Faero.copy()
		# store reference value
		Vtot_zeta=S0.Vtot_zeta.copy()
		Wzeta=S0.Wzeta.copy()
		dZetadt=S0.dZetadt.copy()

		FDstep=1e-6
		Pder_num=np.zeros((Ndim,K,Ndim,K))
		for kk in range(K):
			for dd in range(Ndim):
				# reset grid
				S0.Wzeta=Wzeta.copy()
				S0.dZetadt=dZetadt.copy()
				# Perturb grid
				S0.Wzeta[kk,dd]+=FDstep
				#S0.dZetadt[kk,dd]==FDstep
				# Total velocity
				S0.get_induced_velocity()
				S0.Vtot_zeta=S0.Uzeta+S0.Wzeta-S0.dZetadt+S0.Vind_zeta
				# Force Joukovski at segments
				for nn in range(M):
					S0.Faero[nn,:]=-S0.gamma[nn]*\
					          np.array([-S0.Vtot_zeta[nn,1],S0.Vtot_zeta[nn,0]])
				# Force Joukovski at grid
				Fgrid_here=S0.Faero
				for ff in range(Ndim):
					Pder_num[ff,:,dd,kk]=(Fgrid_here[:,ff]-Fgrid[:,ff])/FDstep

		# check maximum error
		err_max,Eabs,Erel=max_error_tensor(Pder_an,Pder_num)

		tol=5.*FDstep
		assert err_max<tol,\
			       'Maximum error %.2e larger than tolerance %.2e'%(err_max,tol)
		#embed()


	def test_der_Fjouk_dGamma_vtot0(self):
		'''
		Test lin_uvlm2d_sta.solver.der_Fjouk_dGamma_vtot0 against FD.

		@warning: variables should be normalised!
		'''

		S0,Slin=self.Slin.S0,self.Slin
		M,Mw,K=S0.M,S0.Mw,S0.K
		Ndim=2
		S0.nondimvars()

		# redefine circulation
		S0.Gamma=np.array([1.2*(ii-0.7)*(ii+1) for ii in range(M)])
		S0.GammaW=np.array([2.3*(ii-0.3)*(ii+1.5) for ii in range(Mw)])
		S0.gamma=np.dot(S0._TgG,S0.Gamma)
		S0.gammaW=np.dot(S0._TgG_w,S0.GammaW)+np.dot(S0._EgG,S0.Gamma)

		# update induced/total velocity
		S0.get_induced_velocity()
		S0.Vtot_zeta=S0.Uzeta+S0.Wzeta-S0.dZetadt+S0.Vind_zeta

		# Analytical derivative
		Pder_an=Slin.der_Fjouk_dGamma_vtot0()
		Iseg,Icoll=Slin.S0.get_force_matrices()

		# Force Joukovski at segments
		for nn in range(M):
			S0.Faero[nn,:]=-S0.gamma[nn]*\
			                  np.array([-S0.Vtot_zeta[nn,1],S0.Vtot_zeta[nn,0]])
		# Force Joukovski at grid
		Fgrid=S0.Faero.copy()
		# store reference value
		gamma=S0.gamma.copy()
		Gamma=S0.Gamma.copy()

		FDstep=1e-6
		Pder_num=np.zeros((Ndim,K,M))
		for mm in range(M):

			# reset grid
			S0.Gamma=Gamma.copy()
			# Perturb grid
			S0.Gamma[mm]+=FDstep
			# Force Joukovski at segments
			S0.gamma=np.dot(S0._TgG,S0.Gamma)
			S0.Faero[:,:]=0.0
			for nn in range(M):
				S0.Faero[nn,:]=-S0.gamma[nn]*\
				              np.array([-S0.Vtot_zeta[nn,1],S0.Vtot_zeta[nn,0]])
			# Force Joukovski at grid
			Fgrid_here=S0.Faero
			for ff in range(Ndim):
				Pder_num[ff,:,mm]=(Fgrid_here[:,ff]-Fgrid[:,ff])/FDstep

		# check maximum error
		err_max,Eabs,Erel=max_error_tensor(Pder_an,Pder_num)

		tol=5.*FDstep
		assert err_max<tol,\
			       'Maximum error %.2e larger than tolerance %.2e'%(err_max,tol)
		#embed()


	def test_der_Fjouk_dGamma_vind(self):
		'''
		Test lin_uvlm2d_sta.solver.der_Fjouk_dGamma_vind against FD.

		@warning: variables should be normalised!
		'''

		S0,Slin=self.Slin.S0,self.Slin
		M,Mw,K=S0.M,S0.Mw,S0.K
		Ndim=2
		S0.nondimvars()

		# redefine circulation
		S0.Gamma=np.array([1.2*(ii-0.7)*(ii+1) for ii in range(M)])
		S0.GammaW=np.array([2.3*(ii-0.3)*(ii+1.5) for ii in range(Mw)])
		S0.gamma=np.dot(S0._TgG,S0.Gamma)
		S0.gammaW=np.dot(S0._TgG_w,S0.GammaW)+np.dot(S0._EgG,S0.Gamma)

		# update induced/total velocity
		S0.get_induced_velocity()
		Vtot_zeta=S0.Uzeta+S0.Wzeta-S0.dZetadt+S0.Vind_zeta

		# Analytical derivative
		Pder_an,PderW_an=Slin.der_Fjouk_dGamma_vind()
		Iseg,Icoll=Slin.S0.get_force_matrices()

		# Force Joukovski at segments
		for nn in range(M):
			S0.Faero[nn,:]=-S0.gamma[nn]*\
			                        np.array([-Vtot_zeta[nn,1],Vtot_zeta[nn,0]])
		# Force Joukovski at grid
		Fgrid=S0.Faero.copy()
		# store reference value
		gamma=S0.gamma.copy()
		Gamma=S0.Gamma.copy()
		GammaW=S0.GammaW.copy()

		FDstep=1e-6

		# bound term
		Pder_num=np.zeros((Ndim,K,M))
		for mm in range(M):
			# reset circ
			S0.Gamma=Gamma.copy()
			S0.GammaW=GammaW.copy()
			# Perturb circ
			S0.Gamma[mm]+=FDstep

			S0.gamma=np.dot(S0._TgG,S0.Gamma)
			S0.gammaW=np.dot(S0._TgG_w,S0.GammaW)+np.dot(S0._EgG,S0.Gamma)
			#S0.gamma[mm]+=FDstep

			S0.get_induced_velocity()
			S0.Vtot_zeta=S0.Uzeta+S0.Wzeta-S0.dZetadt+S0.Vind_zeta

			for nn in range(M):
				S0.Faero[nn,:]=-gamma[nn]*\
				              np.array([-S0.Vtot_zeta[nn,1],S0.Vtot_zeta[nn,0]])
			# Force Joukovski at grid
			Fgrid_here=S0.Faero
			for ff in range(Ndim):
				Pder_num[ff,:,mm]=(Fgrid_here[:,ff]-Fgrid[:,ff])/FDstep

		# wake term
		PderW_num=np.zeros((Ndim,K,Mw))
		for mm in range(Mw):
			# reset circ
			S0.Gamma=Gamma.copy()
			S0.GammaW=GammaW.copy()
			# Perturb circ
			S0.GammaW[mm]+=FDstep

			S0.gamma=np.dot(S0._TgG,S0.Gamma)
			S0.gammaW=np.dot(S0._TgG_w,S0.GammaW)+np.dot(S0._EgG,S0.Gamma)
			#S0.gammaW[mm]+=FDstep

			S0.get_induced_velocity()
			S0.Vtot_zeta=S0.Uzeta+S0.Wzeta-S0.dZetadt+S0.Vind_zeta

			for nn in range(M):
				S0.Faero[nn,:]=-gamma[nn]*\
				              np.array([-S0.Vtot_zeta[nn,1],S0.Vtot_zeta[nn,0]])
			# Force Joukovski at grid
			Fgrid_here=S0.Faero
			for ff in range(Ndim):
				PderW_num[ff,:,mm]=(Fgrid_here[:,ff]-Fgrid[:,ff])/FDstep


		# check maximum error
		err_max,Eabs,Erel=max_error_tensor(Pder_an,Pder_num)
		errW_max,EabsW,ErelW=max_error_tensor(PderW_an,PderW_num)

		tol=5.*FDstep
		assert err_max<tol,\
			       'Maximum error %.2e larger than tolerance %.2e'%(err_max,tol)
		assert errW_max<tol,\
			 'Wake maximum error %.2e larger than tolerance %.2e'%(errW_max,tol)
		#embed()


class Test_der_dyn(unittest.TestCase):
	'''
	Each method defined in this class contains a test case.
	@warning: by default, only functions whose name starts with 'test' is run.
	'''


	def setUp(self):
		''' Common piece of code to initialise the test '''

		# build reference state
		Mw=4
		M=3
		alpha=0.*np.pi/180.
		ainf=15.0*np.pi/180.
		Uinf=20.*np.array([np.cos(ainf),np.sin(ainf)])
		chord=5.0
		b=0.5*chord
		self.S0=uvlm2d_sta.solver(M,Mw,b,Uinf,alpha,rho=1.225)
		#self.S0.build_flat_plate()
		self.S0.build_camber_plate(Mcamb=50,Pcamb=4)
		self.S0.solve_static_Gamma2d()



	def test_der_Fmass_dZeta(self):
		'''
		Test lin_uvlm2d_dyn.solver.der_Fmass_dZeta against FD. In order
		to compute this term only, the normal vectors are not updated during
		the FD step.

		@warning: variables should be normalised!
		'''


		S0=self.S0
		M,Mw,K=S0.M,S0.Mw,S0.K
		Ndim=2
		# enforce a nonzero reference dGammadt
		for mm in range(M):
			S0.dGammadt[mm]=0.3*(mm+1)


		Slin=lin_uvlm2d_dyn.solver(S0,T=S0.chord/M/S0.Uabs)
		Slin.nondimvars()
		S0.nondimvars()


		### analytical derivatives
		Pder_an=Slin.der_Fmass_dZeta()


		### Numerical derivatives:
		# reference values
		Zeta=S0.Zeta.copy()
		Fmass=np.zeros((M,Ndim))
		DZeta=np.diff(Zeta.T).T
		kcross2D=np.array([[0,-1.],[1,0]])
		Nmat=S0.Nmat
		for nn in range(M):
			Fmass[nn,:]=-np.linalg.norm(DZeta[nn,:])*S0.dGammadt[nn]*Nmat[nn,:]

		# FD
		FDstep=1e-6*S0.b
		Pder_num=0.0*Pder_an
		Fmass_here=0.0*Fmass
		for kk in range(K):
			for dd in range(Ndim):

				# reset grid
				S0.Zeta=Zeta.copy()

				# Perturb grid
				S0.Zeta[kk,dd]+=FDstep
				if kk==S0.K-1:
					S0.ZetaW[0,dd]+=FDstep
				# update normal and area matrices
				S0.get_Nmat()
				DZeta=np.diff(S0.Zeta.T).T
				for nn in range(M):
					Fmass_here[nn,:]=-np.linalg.norm(DZeta[nn,:])*\
					                               S0.dGammadt[nn]*S0.Nmat[nn,:]

		        # get derivative
				Pder_num[0,:,dd,kk]=(Fmass_here[:,0]-Fmass[:,0])/FDstep
				Pder_num[1,:,dd,kk]=(Fmass_here[:,1]-Fmass[:,1])/FDstep

		# check maximum error
		err_max,Eabs,Erel=max_error_tensor(Pder_an,Pder_num)

		tol=5.*FDstep
		assert err_max<tol,\
			       'Maximum error %.2e larger than tolerance %.2e'%(err_max,tol)
		#embed()


	def test_der_Fmass_dGammadt(self):
		'''
		Test lin_uvlm2d_dyn.solver.der_Fmass_dZeta against FD. In order
		to compute this term only, the normal vectors are not updated during
		the FD step.

		@warning: variables should be normalised!
		'''


		S0=self.S0
		M,Mw,K=S0.M,S0.Mw,S0.K
		Ndim=2
		# enforce a nonzero reference dGammadt
		for mm in range(M):
			S0.dGammadt[mm]=0.3*(mm+1)

		Slin=lin_uvlm2d_dyn.solver(S0,T=S0.chord/M/S0.Uabs)
		Slin.nondimvars()
		S0.nondimvars()

		### analytical derivatives
		Pder_an=Slin.der_Fmass_dGammadt()

		### Numerical derivatives:

		# reference values
		dGammadt=S0.dGammadt.copy()
		DZeta=np.diff(S0.Zeta.T).T
		kcross2D=np.array([[0,-1.],[1,0]])
		Fmass=np.zeros((M,Ndim))
		for nn in range(M):
			Fmass[nn,:]=-np.linalg.norm(DZeta[nn,:])*S0.dGammadt[nn]*S0.Nmat[nn,:]

		# FD
		FDstep=1e-6*S0.b
		Pder_num=0.0*Pder_an
		Fmass_here=0.0*Fmass

		for mm in range(M):

				# reset grid
				S0.dGammadt=dGammadt.copy()

				# Perturb grid
				S0.dGammadt[mm]+=FDstep
				for nn in range(M):
					Fmass_here[nn,:]=-np.linalg.norm(DZeta[nn,:])*\
					                               S0.dGammadt[nn]*S0.Nmat[nn,:]

		        # get derivative
				Pder_num[0,mm,:]=(Fmass_here[:,0]-Fmass[:,0])/FDstep
				Pder_num[1,mm,:]=(Fmass_here[:,1]-Fmass[:,1])/FDstep


		# check maximum error
		err_max,Eabs,Erel=max_error_tensor(Pder_an,Pder_num)

		tol=5.*FDstep
		assert err_max<tol,\
			       'Maximum error %.2e larger than tolerance %.2e'%(err_max,tol)
		#embed()



if __name__=='__main__':

	### Static terms
	#T=Test_der_sta()
	#T.setUp()
	unittest.main()

	## Induced velocity
	#T.test_der_WncV_dZeta()
	#T.test_der_Wnc0AGamma_dZeta()

	## Joukovski force
	#T.test_der_Fjouk_dV()
	#T.test_der_Fjouk_dGamma_vtot0()
	#T.test_der_Fjouk_dGamma_vind()
	#T.test_der_Fjouk_dZeta_vind_all()


	### Dynamic terms
	#T=Test_der_dyn()
	#T.setUp()
	#unittest.main()

	# Added mass
	#T.test_der_Fmass_dZeta()
	#T.test_der_Fmass_dGammadt()
