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
		###self.THWzetaW=np.zeros((self.NT,self.Kw,Ndim))# gust over wake - not in linear eq.s

		# Output
		self.THFaero=np.zeros((self.NT,Ndim))		  # total aerodynamic force
		self.THFaero_m=np.zeros((self.NT,Ndim))		  # total added mass force
		self.THFdistr=np.zeros((self.NT,K,Ndim)) # force distribution on grid

		# # Other
		# self.THVtot_zeta=np.zeros((self.NT,self.K,Ndim))
		#
		# # instanteneous velocities at wake
		# self.WzetaW=np.zeros((self.Kw,Ndim))   # gust
		# self.UzetaW = np.zeros((self.Kw,Ndim)) # background
		# for nn in range(self.Kw): 
		# 	self.UzetaW[nn,:]=self.Uinf

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
		self.SolDyn.Output.drop(THFaero=self.THFaero,THFaero_m=self.THFaero_m,
			                                             THFdistr=self.THFdistr)



		'''
		Params=[S0.M,S0.Mw,S0.K,S0.Kw,
		        S0.b,S0.alpha,S0.Uinf,S0.rho,S0.qinf,
				S0.perc_ring,S0.perc_coll,S0.perc_interp,
		'''



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



	def get_force_matrices(self):
		''' Produce matrices to interpolate the force from segments and 
		collocation points to grid points '''


		M,K=self.S0.M,self.S0.K
		Iseg=np.zeros((K,M))
		Icoll=np.zeros((K,M))

		Iseg[:M,:]=np.eye(M)
		#Icoll
		wvcoll=np.zeros((K,))
		wvcoll[0]=1.-self.S0.perc_interp 
		wvcoll[1]=self.S0.perc_interp 
		Icoll=scalg.circulant(wvcoll)[:,:M]

		return Iseg, Icoll



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

		### define constant matrices
		# convection wake intensity
		Cgamma,CgammaW=self.get_Cgamma_matrices()
		# convection wake coordinates
		Czeta,CzetaW=self.get_Czeta_matrices()
		# force interpolation
		Iseg,Icoll=self.get_force_matrices()


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
			self.FmatSta[:,:]=0.0


		### store t=0 state/output: 
		# @warning: dGammadt=0 at t=0 if starting from steady solution
		self.THgamma[0,:]=self.gamma
		self.THgammaW[0,:]=self.gammaW
		self.THGamma[0,:]=self.Gamma
		self.THGammaW[0,:]=self.GammaW
		self.THdGammadt[0,:]=self.dGammadt 
		self.THFaero[0,:]=self.FmatSta.sum(0)		

		self.Xnew=[]

		##### State space system:
		# coordinates reshaped in Fortran order, with first index changing 
		# faster. In the reshaped array, the first K dof are the x component.
		# @warning: order of input u different from Ref.[2]

		Nx=2*M+Mw
		Nu=6*K

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
		LU,P=scalg.lu_factor(Ess)
		Ass=scalg.lu_solve( (LU,P), Fss)
		Bss=scalg.lu_solve( (LU,P), Gss)

		#Ess1=Ess[:M,:]
		#Fss1=Fss[:M,:]
		#Gss1=Gss[:M,:]
		#Ass1=Ass[:M,:]
		#Bss1=Bss[:M,:]


		# Output matrices - To be completed
		Css=np.eye(Nx)
		Dss=np.zeros((Nx,Nu))


		### Time-stepping
		xold=np.concatenate( (self.Gamma,self.GammaW,self.dGammadt) )

		for tt in range(1,NT):
			print('step: %.6d of %.6d'%(tt,NT))

			# define new input
			unew=np.concatenate((
				self.THZeta[tt,:,:].reshape((2*K,),order='F'),
						self.THdZetadt[tt,:,:].reshape((2*K,),order='F'),
								self.THWzeta[tt,:,:].reshape((2*K,),order='F')))

			# get new state
			xnew=np.dot(Ass,xold)+np.dot(Bss,unew)
			# xnew_test=np.linalg.solve(Ess,np.dot(Fss,xold))+\
			# 							   np.linalg.solve(Ess,np.dot(Gss,unew))
			#gnew=np.dot(Ass1,xold)+np.dot(Bss1,unew)
			#if tt<4: embed()
			xold = xnew.copy()
			#self.Xnew.append(xnew.copy())

			# store state
			self.THGamma[tt,:]=xnew[:M]
			self.THGammaW[tt,:]=xnew[M:M+Mw]
			self.THdGammadt[tt,:]=xnew[M+Mw:]
			#embed()

			# # Produce gamma
			self.THgamma[tt,:]=np.dot(S0._TgG,self.THGamma[tt,:])
			self.THgammaW[tt,:]=np.dot(S0._TgG_w,self.THGammaW[tt,:])\
											 +np.dot(S0._EgG,self.THGamma[tt,:])


			### compute force

			# Total Vorticity
			self.GammaTot=S0.Gamma+self.THGamma[tt,:]
			self.GammaWTot=S0.GammaW+self.THGammaW[tt,:]
			self.gammaTot=S0.gamma+self.THgamma[tt,:]
			self.gammaWTot=S0.gammaW+self.THgammaW[tt,:]


			# Total deformations (wake only changed at bound/wake interface)
			self.ZetaTot=S0.Zeta+self.THZeta[tt,:,:]
			self.ZetaWTot=S0.ZetaW.copy()
			self.ZetaWTot[S0.Map_bw[:,1],:]=self.ZetaTot[S0.Map_bw[:,0],:] 

			# induced velocity
			self.get_total_induced_velocity()

			# total velocity
			self.Vtot_zeta=S0.Uzeta+S0.Wzeta-S0.dZetadt\
				  +self.THWzeta[tt,:,:]-self.THdZetadt[tt,:,:]+self.VindTot_zeta

			# Force - Joukovski
			for nn in range(M):
				Fjouk[nn,:]=-self.gammaTot[nn]*\
				          np.array([-self.Vtot_zeta[nn,1],self.Vtot_zeta[nn,0]])
			# Added mass - collocation points
			DZeta=np.diff(self.ZetaTot.T).T
			kcross2D=np.array([[0,-1.],[1,0]])
			self.Nmat=0.0*S0.Nmat
			for ii in range(K-1):
				self.Nmat[ii,:]=np.dot(kcross2D,DZeta[ii,:])/\
				                                     np.linalg.norm(DZeta[ii,:])
			for nn in range(M):
				Fmass[nn,:]=-np.linalg.norm(DZeta[nn,:])*\
				                          self.THdGammadt[tt,nn]*self.Nmat[nn,:]


			# interpolate force over grid
			Faero_m=np.dot(Icoll,Fmass)
			self.Fmat=np.dot(Iseg,Fjouk)+Faero_m

			# store output
			self.THFaero[tt,:]=self.Fmat.sum(0)
			self.THFaero_m[tt,:]=Faero_m.sum(0)
			self.THFdistr[tt,:,:]=self.Fmat
			#self.THVtot_zeta[tt,:,:]=self.Vtot_zeta

		# terminate
		self._exec_time=time.time()-start_time
		print('Done in %.1f sec!' % self._exec_time)

		# dimensionalise
		self.dimvars()
		S0.dimvars()


		# save:
		if self._saveout:
			save.h5file(self._savedir,self._savename, *(S0.SolSta,self.SolDyn,))


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
	Mw=4
	M=3
	alpha=2.*np.pi/180.
	ainf=2.0*np.pi/180.
	Uinf=20.*np.array([np.cos(ainf),np.sin(ainf)])
	chord=3.
	b=0.5*chord

	# static exact solution
	S0=uvlm2d_sta.solver(M,Mw,b,Uinf,alpha,rho=1.225)
	S0.build_camber_plate(Mcamb=10,Pcamb=4)
	S0.solve_static_Gamma2d()

	# linearise dynamic
	T=2.0
	Slin=solver(S0,T)
	#Slin.solve_dyn_Gamma2d()