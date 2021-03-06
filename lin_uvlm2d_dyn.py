'''
2D linearised UVLM solver
author: S. Maraniello
date: 19 Jul 2017

Nomenclature as per "State space relatisation of potential flow unsteady 
aerodynamics with arbitrary kinematics

Ref.[1]: Anderson, Fundamentals of Aerodynamics
Ref.[2]: Simpson, Palacios and Maraniello, Scitech 2017
Ref.[3]: Katz and Plotkin, Low speed aerodynamics
'''

import scipy as sc
import scipy.linalg as scalg
import scipy.signal as scsig
import scipy.sparse as sparse
import numpy as np
import multiprocessing as mpr
import matplotlib.pyplot as plt
from IPython import embed
import save
import time

import lin_uvlm2d_sta
from uvlm2d_sta import biot_savart_2d
import libder


class solver(lin_uvlm2d_sta.solver):
	'''
	Linearised dyamic solver.
	'''


	def __init__(self,S0,T):
		''' 
		Inherit methods from static solver. Initialisation requires an instance 
		of the static UVLM 2D solver, S0. 
		Quantities have the same nomenclature as nonlinear solver, but have
		to be intended as deltas w.r.t. the reference configuration in S0
		'''

		super().__init__(S0)
		Ndim=2

		# create mapping for derivatives computation
		self.S0.mapping()

		### initialise Delta terms
		M,Mw,K,Kw=S0.M,S0.Mw,S0.K,S0.Kw

		# dynamics parameters
		dl=2.*S0.b/M
		self.dt=dl/S0.Uabs
		self.T=T
		self.time=np.arange(0.,self.T,self.dt)
		self.NT=len(self.time)


		### dynamics arrays

		# State
		self.THGamma=np.zeros((self.NT,M))
		self.THGammaW=np.zeros((self.NT,Mw))
		self.THdGammadt=np.zeros((self.NT,M,))
		self.THgamma=np.zeros((self.NT,M))
		self.THgammaW=np.zeros((self.NT,Mw))

		# Input
		self.THZeta=np.zeros((self.NT,K,Ndim))    # aerofoil pos.
		self.THdZetadt=np.zeros((self.NT,K,Ndim)) # aerofoil vel.
		self.THWzeta=np.zeros((self.NT,K,Ndim))   # gust over aerofoil
		###self.THWzetaW=np.zeros((self.NT,self.Kw,Ndim))# unecessary-frozen wake

		# Output
		self.THFaero=np.zeros((self.NT,Ndim))		  # total aerodynamic force
		self.THFaero_m=np.zeros((self.NT,Ndim))		  # total added mass force
		self.THFdistr=np.zeros((self.NT,K,Ndim)) # force distribution on grid

		# # Other
		# self.THVtot_zeta=np.zeros((self.NT,self.K,Ndim))

		# Hall's correction: 
		# regulates the dissipation of last vortex: 
		# 1.0=full dissipation (i.e. no correction)
		self.eps_Hall=1.0 

		# flags
		self._savewake=True
		self._imp_start=False
		#savename
		self._saveout=True
		self._savedir='./res/'
		self._savename='dummy_lindyn.h5'
		self._print=True
		self._sparseSS=True # produces SS system with sparse matrices

		# Prepare Input/Output classes
		self.SolDyn=save.Output('sollindyn')
		self.SolDyn.Param=save.Output('params')
		self.SolDyn.Input=save.Output('input')
		self.SolDyn.State=save.Output('state')
		self.SolDyn.Output=save.Output('output')


	def get_Cgamma_matrices(self):
		''' Produce matrices for propagation of wake - modified Ref.[4] '''

		M,Mw=self.S0.M,self.S0.Mw
		Cgamma=np.zeros((Mw,M))
		CgammaW=np.zeros((Mw,Mw))

		# 1st element of wake = last element of aerofoil
		Cgamma[0,-1]=1.0

		# wake shift: holds only if Cgamma[0,-1]=1.0
		for ii in range(1,Mw): CgammaW[ii,ii-1]=1.0

		### Hall's correction
		CgammaW[-1,-2]=self.eps_Hall
		CgammaW[-1,-1]=1.-self.eps_Hall

		return Cgamma, CgammaW


	def get_Czeta_matrices(self):
		''' Produce matrices for propagation of wake - eq.(2) Ref.[3] '''

		K,Kw=self.S0.K,self.S0.Kw
		Czeta=np.zeros((Kw,K))
		CzetaW=np.zeros((Kw,Kw))

		# 1st element of wake = last element of aerofoil
		Czeta[0,-1]=1.0

		# wake shift
		for ii in range(1,Kw): CzetaW[ii,ii-1]=1.0

		return Czeta, CzetaW


	def solve_dyn_Gamma2d(self):
		'''
		In this method, the wake is assumed to be frozen (for the purpose of 
		of linearisation the wake is frozen)

		This solution imposes the steady kutta condition at the trailing edge,
		namely: 
			self.Gamma[-1]=self.GammaW[0]
		or also
			self.gammaW[0]=0
		at each time-step
		'''

		start_time = time.time()
		S0=self.S0

		# initialise
		Ndim=2
		M, Mw=S0.M, S0.Mw
		K, Kw=S0.K, S0.Kw
		NT=self.NT

		# Force 
		Fjouk=np.zeros((M,Ndim))  # at segments (grid in 2D)
		Fmass=np.zeros((M,Ndim))  # at collocation points


		### steady solution
		# This will update things only if the input of the linear static model
		# (self.Zeta, self.dZetadt, self.Wzeta) are initialised to be non-zero.

		# asset input at t=0 in TH* arrays are as in linear static solver
		assert np.max(np.abs((self.THZeta[0,:,:]-self.Zeta)))<1e-8 and\
			   np.max(np.abs((self.THdZetadt[0,:,:]-self.dZetadt)))<1e-8 and\
			   np.max(np.abs((self.THWzeta[0,:,:]-self.Wzeta)))<1e-8,\
		                   'Input of linear static/dynamic solver not matching!'

              
		# solve steady
		self.solve_static_Gamma2d()
		# nondimensionalise
		self.nondimvars()
		S0.nondimvars()

		if self._imp_start: 
			self.Gamma[:]=0.
			self.gamma[:]=0.
			self.GammaW[:]=0.
			self.gammaW[:]=0.
			self.Faero[:,:]=0.0


		### store t=0 state/output: 
		# @warning: dGammadt=0 at t=0 if starting from steady solution
		self.THgamma[0,:]=self.gamma
		self.THgammaW[0,:]=self.gammaW
		self.THGamma[0,:]=self.Gamma
		self.THGammaW[0,:]=self.GammaW
		self.THdGammadt[0,:]=self.dGammadt 
		self.THFaero[0,:]=self.Faero.sum(0)		
		self.THFdistr[0,:,:]=self.Faero.copy()

		### State space system
		Ass,Bss,Css_j,Dss_j,Css_m,Dss_m=self.build_ss()

		# define update methods
		if self._sparseSS:
			def get_xnew(Ass,Bss,xold,unew):
				return  Ass.dot(xold)+Bss.dot(unew)
			def get_ynew(Css_m,Dss_m,Css_j,Dss_j,xnew,unew):
				ynew=Css_m.dot(xnew)+Dss_m.dot(unew)
				Faero_m=ynew.reshape((K,2),order='F')
				ynew=ynew+Css_j.dot(xnew)+Dss_j.dot(unew)
				Faero=ynew.reshape((K,2),order='F')
				return ynew,Faero_m,Faero	
		else:
			def get_xnew(Ass,Bss,xold,unew):
				return np.dot(Ass,xold)+np.dot(Bss,unew)
			def get_ynew(Css_m,Dss_m,Css_j,Dss_j,xnew,unew):
				ynew=np.dot(Css_m,xnew)+np.dot(Dss_m,unew)
				Faero_m=ynew.reshape((K,2),order='F')
				ynew=ynew+np.dot(Css_j,xnew)+np.dot(Dss_j,unew)
				Faero=ynew.reshape((K,2),order='F')
				return ynew,Faero_m,Faero


		print('Time-stepping started...')
		t0=time.time()
		### Time-stepping
		xold=np.concatenate( (self.Gamma,self.GammaW,self.dGammadt) )

		for tt in range(1,NT):
			if self._print:
				print('step: %.6d of %.6d'%(tt+1,NT))

			# define new input
			unew=np.concatenate((
				self.THZeta[tt,:,:].reshape((2*K,),order='F'),
						self.THdZetadt[tt,:,:].reshape((2*K,),order='F'),
								self.THWzeta[tt,:,:].reshape((2*K,),order='F')))

			# get new state
			#xnew=np.dot(Ass,xold)+np.dot(Bss,unew)
			xnew=get_xnew(Ass,Bss,xold,unew)
			xold=xnew

			# store state
			self.THGamma[tt,:]=xnew[:M]
			self.THGammaW[tt,:]=xnew[M:M+Mw]
			self.THdGammadt[tt,:]=xnew[M+Mw:]
			#embed()

			# # Produce gamma
			self.THgamma[tt,:]=np.dot(S0._TgG,self.THGamma[tt,:])
			self.THgammaW[tt,:]=np.dot(S0._TgG_w,self.THGammaW[tt,:])\
											 +np.dot(S0._EgG,self.THGamma[tt,:])

			### Linearised output
			# ynew=np.dot(Css_m,xnew)+np.dot(Dss_m,unew)
			# Faero_m=ynew.reshape((K,2),order='F')
			# ynew=ynew+np.dot(Css_j,xnew)+np.dot(Dss_j,unew)
			# Faero=ynew.reshape((K,2),order='F')
			ynew,Faero_m,Faero=get_ynew(Css_m,Dss_m,Css_j,Dss_j,xnew,unew)

			# store output
			self.THFaero[tt,:]=Faero.sum(0)
			self.THFaero_m[tt,:]=Faero_m.sum(0)
			self.THFdistr[tt,:,:]=Faero
			#self.THVtot_zeta[tt,:,:]=self.Vtot_zeta

		tend=time.time()-t0
		print('\tDone in %.2f sec!' %tend)

		# terminate
		self._exec_time=time.time()-start_time
		print('Dynamic solution done in %.1f sec!' % self._exec_time)

		# dimensionalise
		self.dimvars()
		S0.dimvars()

		# save:
		if self._saveout:

			# Prepare Input/Output classes
			self.SolDyn=save.Output('sollindyn')
			self.SolDyn.Param=save.Output('params')
			self.SolDyn.Param.drop(dt=self.dt,T=self.T,time=self.time,
										      NT=self.NT,eps_Hall=self.eps_Hall)
			self.SolDyn.Input=save.Output('input')
			self.SolDyn.Input.drop(THZeta=self.THZeta,
				                  THdZetadt=self.THdZetadt,THWzeta=self.THWzeta)
			self.SolDyn.State=save.Output('state')
			self.SolDyn.State.drop(THGamma=self.THGamma,THGammaW=self.THGammaW,
				                                     THdGammadt=self.THdGammadt)
			self.SolDyn.Output=save.Output('output')
			self.SolDyn.Output.drop(THFaero=self.THFaero,
				                THFaero_m=self.THFaero_m,THFdistr=self.THFdistr)

			save.h5file(self._savedir,self._savename, *(S0.SolSta,self.SolDyn,))


	def build_ss(self,AllMats=False):
		'''
		Build state-space matrices in the form
			E x^{n+1} = F x^n + G u^{n+1}
			y^{n+1} = C x^{n+1} + D u^{n+1}

		@note: the E,F,G,C,D matrices are built starting from full matrices. The
		allocation of tensors (in lin_uvlm2d_sta module) is expensive but not
		because sparsity is not exploited. Rather, this is due to the foor loops
		over multi-dimensional arrays and non-contiguous elements of the array.
		The conversion to sparse matrix, however:
			1. is fast, hence no need to define matrices directly as sparse
			2. allows faster time-stepping
			3. reduces memory usage.
		'''

		t0=time.time()
		print('build_ss started...')

		S0=self.S0
		M,Mw=S0.M,S0.Mw
		K,Kw=S0.K,S0.Kw

		Nx=2*M+Mw
		Nu=6*K

		### define constant matrices
		# convection wake intensity
		Cgamma,CgammaW=self.get_Cgamma_matrices()
		## convection wake coordinates
		#Czeta,CzetaW=self.get_Czeta_matrices()
		# force interpolation
		Iseg,Icoll=S0.get_force_matrices()

		### State Matrices
		Ess,Fss=np.zeros((Nx,Nx)),np.zeros((Nx,Nx))

		Ess[:M,:M]=S0.A
		Ess[:M,M:M+Mw]=S0.AW
		Ess[M:M+Mw,M:M+Mw]=np.eye(Mw)
		Ess[M+Mw:2*M+Mw,:M]=-np.eye(M)
		Ess[M+Mw:2*M+Mw,M+Mw:2*M+Mw]=self.dt*np.eye(M)

		Fss[M:M+Mw,:M]=Cgamma
		Fss[M:M+Mw,M:M+Mw]=CgammaW
		Fss[M+Mw:2*M+Mw,:M]=-np.eye(M)

		### Input matrix
		W0=np.zeros((M,2*K))
		W0[:,:K]=np.dot(S0._Wnc[0,:,:],S0._Wcv)
		W0[:,K:2*K]=np.dot(S0._Wnc[1,:,:],S0._Wcv)

		Gss=np.zeros((Nx,Nu))
		Gss[:M,:K]=-self.dVcoll_dZeta[0,:,:]
		Gss[:M,K:2*K]=-self.dVcoll_dZeta[1,:,:]
		Gss[:M,2*K:4*K]=W0
		Gss[:M,4*K:6*K]=-W0

		### State space description
		if self._sparseSS:
			E=sparse.bsr_matrix(Ess)
			invE=sparse.linalg.splu(E)
			Ass=invE.solve(Fss)
			Bss=invE.solve(Gss)
		else:
			LU,P=scalg.lu_factor(Ess)
			Ass=scalg.lu_solve( (LU,P), Fss)
			Bss=scalg.lu_solve( (LU,P), Gss)

		### Output matrices - To be completed
		Ny=2*K
		Css_m=np.zeros((Ny,Nx))
		Dss_m=np.zeros((Ny,Nu))
		Css_j=np.zeros((Ny,Nx))
		Dss_j=np.zeros((Ny,Nu))
		

		### circulatory terms
		Css_j[:K,:M]=self.DFj_dGamma[0,:,:]
		Css_j[K:,:M]=self.DFj_dGamma[1,:,:]
		Css_j[:K,M:M+Mw]=self.DFj_dGammaW[0,:,:]
		Css_j[K:,M:M+Mw]=self.DFj_dGammaW[1,:,:]

		Dss_j[:K,:K]   =self.DFj_dZeta[0,:,0,:]
		Dss_j[:K,K:2*K]=self.DFj_dZeta[0,:,1,:]
		Dss_j[K:,:K]   =self.DFj_dZeta[1,:,0,:]
		Dss_j[K:,K:2*K]=self.DFj_dZeta[1,:,1,:]

		Dss_j[:K,2*K:3*K]=-self.DFj_dV[0,:,0,:]
		Dss_j[:K,3*K:4*K]=-self.DFj_dV[0,:,1,:]
		Dss_j[K:,2*K:3*K]=-self.DFj_dV[1,:,0,:]
		Dss_j[K:,3*K:4*K]=-self.DFj_dV[1,:,1,:]		

		Dss_j[:K,4*K:5*K]=self.DFj_dV[0,:,0,:]
		Dss_j[:K,5*K:6*K]=self.DFj_dV[0,:,1,:]
		Dss_j[K:,4*K:5*K]=self.DFj_dV[1,:,0,:]
		Dss_j[K:,5*K:6*K]=self.DFj_dV[1,:,1,:]	

		### added mass terms
		# der Fmass w.r.t. dGammadt
		DFm_dGammadt=self.der_Fmass_dGammadt()
		Css_m[:K,M+Mw:2*M+Mw]=np.dot(Icoll,DFm_dGammadt[0,:,:]) # force x
		Css_m[K:,M+Mw:2*M+Mw]=np.dot(Icoll,DFm_dGammadt[1,:,:]) # force y
		# der Fmass w.r.t. Zeta
		DFm_dZeta=self.der_Fmass_dZeta()
		Dss_m[:K,:K]=np.dot(Icoll,DFm_dZeta[0,:,0,:]) # force x w.r.t. x displ.
		Dss_m[:K,K:2*K]=np.dot(Icoll,DFm_dZeta[0,:,1,:]) # force x w.r.t. y displ.
		Dss_m[K:,:K]=np.dot(Icoll,DFm_dZeta[1,:,0,:]) # force y w.r.t. x displ.
		Dss_m[K:,K:2*K]=np.dot(Icoll,DFm_dZeta[1,:,1,:]) # force y w.r.t. y displ.

		# convert to sparse
		if self._sparseSS:
			Ass=sparse.bsr_matrix(Ass)
			Bss=sparse.bsr_matrix(Bss)
			Css_m=sparse.bsr_matrix(Css_m)
			Css_j=sparse.bsr_matrix(Css_j) # full matrix but BSR not penalising
			Dss_m=sparse.bsr_matrix(Dss_m)
			Dss_j=sparse.bsr_matrix(Dss_j)
			#embed()

		if AllMats: outs=(Ass,Bss,Css_j,Dss_j,Css_m,Dss_m,Ess,Fss,Gss)
		else: outs=(Ass,Bss,Css_j,Dss_j,Css_m,Dss_m)

		tend=time.time()-t0
		print('\tDone in %.2f sec!' %tend)

		return outs



	def der_Fmass_dZeta(self):
		'''
		Derivative of added mass force at the collocation point w.r.t. changes
		in grid geometry.

		In 2D problems, this matrix is a sparse/constant matrix which only
		depends of the dGammadt_0, and not on the vortex rings coordinates.
		Importantly, this term is always zero when starting from a steady 
		solution, as self.S0.dGammadt=0
		'''

		Ndim=2
		K=self.S0.K
		M=self.S0.M
		Mw=self.S0.Mw

		Der=np.zeros((Ndim,M,Ndim,K))

		# In 2D the derivative does not depend on the vortex grid coordinates
		zeta01_dummy=np.array([0.,0.])
		zeta02_dummy=np.array([0.,0.])
		
		DerLocal=libder.der_NormalArea_dZeta(
							 zeta01=zeta01_dummy,zeta02=zeta02_dummy,dGamma=1.0)

		# loop collocation points
		for mm in range(M):

			# identify nodes connected to collocation point
			map_cv=self.S0.Map_cv[mm,:]

			### extract vertices of ring with collocation point
			# not required, as independent on coordinates in 2D
			#zeta_a=self.S0.Zeta[map_cv[0],:]
			#zeta_b=self.S0.Zeta[map_cv[1],:]

			# Local derivatives
			DerLocalHere=self.S0.dGammadt[mm]*DerLocal

			# Assembly
			Der[:,mm,0,map_cv]=Der[:,mm,0,map_cv]+DerLocalHere[:,:2]
			Der[:,mm,1,map_cv]=Der[:,mm,1,map_cv]+DerLocalHere[:,2:]

		return Der



	def der_Fmass_dGammadt(self):
		'''
		Derivative of added mass force at the collocation point w.r.t. changes
		in dGammadt.
		'''

		Ndim=2
		K=self.S0.K
		M=self.S0.M
		Mw=self.S0.Mw

		Der=np.zeros((Ndim,M,M))

		# loop collocation points
		for mm in range(M):

			# identify nodes connected to collocation point
			map_cv=self.S0.Map_cv[mm,:]

			### extract vertices of ring with collocation point
			# not required, as independent on coordinates in 2D
			zeta_a=self.S0.Zeta[map_cv[0],:]
			zeta_b=self.S0.Zeta[map_cv[1],:]
			Area=np.linalg.norm(zeta_b-zeta_a)
			Norm=self.S0.Nmat[mm,:]

			# Assembly
			Der[:,mm,mm]=Der[:,mm,mm]-Area*Norm

		return Der



	def nondimvars(self):
		'''
		Nondimensionalise static and dynamic variables
		'''

		# static variables
		super().nondimvars()

		# pointer to self.S0
		S0=self.S0

		### time
		self.dt=self.dt/S0.tref
		self.time=self.time/S0.tref

		# State
		self.THGamma=self.THGamma/S0.gref
		self.THGammaW=self.THGammaW/S0.gref
		self.THdGammadt=self.THdGammadt/(S0.gref/S0.tref)
		self.THgamma=self.THgamma/S0.gref
		self.THgammaW=self.THgammaW/S0.gref
		
		# Input
		self.THZeta=self.THZeta/S0.b
		self.THdZetadt=self.THdZetadt/S0.Uabs
		self.THWzeta=self.THWzeta/S0.Uabs

		# Output
		self.THFaero=self.THFaero/S0.Fref		  
		self.THFaero_m=self.THFaero_m/S0.Fref
		self.THFdistr=self.THFdistr/S0.Fref

		### wake
		# self.WzetaW=self.WzetaW/self.Uabs
		# self.UzetaW=self.UzetaW/self.Uabs


	def dimvars(self):
		'''
		Dimensionalise variables of solver.
		'''

		# static variables
		super().dimvars()

		# pointer to self.S0
		S0=self.S0

		### time
		self.dt=self.dt*S0.tref
		self.time=self.time*S0.tref

		# State
		self.THGamma=self.THGamma*S0.gref
		self.THGammaW=self.THGammaW*S0.gref
		self.THdGammadt=self.THdGammadt*(S0.gref/S0.tref)
		self.THgamma=self.THgamma*S0.gref
		self.THgammaW=self.THgammaW*S0.gref
		
		# Input
		self.THZeta=self.THZeta*S0.b
		self.THdZetadt=self.THdZetadt*S0.Uabs
		self.THWzeta=self.THWzeta*S0.Uabs

		# Output
		self.THFaero=self.THFaero*S0.Fref		  
		self.THFaero_m=self.THFaero_m*S0.Fref
		self.THFdistr=self.THFdistr*S0.Fref

		### wake
		# self.WzetaW=self.WzetaW*self.Uabs
		# self.UzetaW=self.UzetaW*self.Uabs




if __name__=='__main__':

	import time
	import uvlm2d_sta
	import pp_uvlm2d as pp

	### build reference state
	Mw=8
	M=3
	alpha=2.*np.pi/180.
	ainf=2.0*np.pi/180.
	Uinf=20.*np.array([np.cos(ainf),np.sin(ainf)])
	chord=3.
	b=0.5*chord

	# static exact solution
	S0=uvlm2d_sta.solver(M,Mw,b,Uinf,alpha,rho=1.225)
	S0.build_camber_plate(Mcamb=30,Pcamb=4)
	S0.solve_static_Gamma2d()

	# linearise dynamic
	T=2.0
	Slin=solver(S0,T)
	Slin.solve_dyn_Gamma2d()