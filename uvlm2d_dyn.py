'''
2D UVLM solver

Nomenclature as per "State space relatisation of p[otential flow unsteady 
aerodynamics with arbitrary kinematics

Ref.[1]: Anderson, Fundamentals of Aerodynamics
Ref.[2]: Simpson, Palacios and Maraniello, Scitech 2017
Ref.[3]: Katz and Plotkin, Low speed aerodynamics
Ref.[4]: Hall, AIAA Journal, 1994

'''

import time
import scipy as sc
import scipy.linalg as scalg
import numpy as np
import multiprocessing as mpr
import matplotlib.pyplot as plt
from IPython import embed

import uvlm2d_sta
from uvlm2d_sta import biot_savart_2d
import save


class solver(uvlm2d_sta.solver):
	'''
	Inherit methods of static solver
	'''

	def __init__(self,T,M,Mw,b,Uinf,alpha,rho=1.225):

		# allocate variables for static
		super().__init__(M,Mw,b,Uinf,alpha,rho)
		Ndim=2

		# dynamics parameters
		dl=2.*self.b/self.M
		self.dt=dl/self.Uabs
		self.T=T
		self.time=np.arange(0.,self.T,self.dt)
		self.NT=len(self.time)
		self.tref=self.b/self.Uabs

		# dynamics arrays
		# @note: the velocity of the bound grid is computed by backward difference
		# on the time history of the aerofoil position, self.THZeta
		self.THZeta=np.zeros((self.NT,self.K,Ndim))   # aerofoil 
		self.THZetaW=np.zeros((self.NT,self.Kw,Ndim)) # wake
		self.THWzeta=np.zeros((self.NT,self.K,Ndim))  # gust over aerofoil
		self.THWzetaW=np.zeros((self.NT,self.Kw,Ndim))# gust over wake
		self.THFaero=np.zeros((self.NT,Ndim))		  # total aerodynamic force
		self.THFaero_m=np.zeros((self.NT,Ndim))		  # total added mass force
		self.THFdistr=np.zeros((self.NT,self.K,Ndim)) # force distribution on grid
		self.THVtot_zeta=np.zeros((self.NT,self.K,Ndim)) # induced velocity at zeta

		# state time-histories
		self.THgamma=np.zeros((self.NT,self.M))
		self.THgammaW=np.zeros((self.NT,self.Mw))
		self.THGamma=np.zeros((self.NT,self.M))
		self.THGammaW=np.zeros((self.NT,self.Mw))

		# instanteneous velocities at wake
		self.WzetaW=np.zeros((self.Kw,Ndim))   # gust
		self.UzetaW = np.zeros((self.Kw,Ndim)) # background
		for nn in range(self.Kw): 
			self.UzetaW[nn,:]=self.Uinf

		# Hall's correction: 
		# regulates the dissipation of last vortex: 
		# 1.0=full dissipation (i.e. no correction)
		self.eps_Hall=1.0 

		# flags
		self._savewake=True
		self._quasi_steady=False
		self._update_AIC=True
		self._imp_start=False

		#savename
		self._savedir='./res/'
		self._savename='dummy.h5'



	def nondimvars(self):
		'''
		Nondimensionalise static and dynamic variables
		'''

		# static variables
		super().nondimvars()

		# time
		self.dt=self.dt/self.tref
		self.time=self.time/self.tref

		# wake
		self.WzetaW=self.WzetaW/self.Uabs
		self.UzetaW=self.UzetaW/self.Uabs

		# prescribed time histories
		self.THWzeta=self.THWzeta/self.Uabs
		self.THWzetaW=self.THWzetaW/self.Uabs
		self.THZeta=self.THZeta/self.b 
		self.THZetaW=self.THZetaW/self.b #<-- never prescribed


	def dimvars(self):
		'''
		Nondimensionalise static and dynamic variables
		'''

		# static variables
		super().dimvars()

		# time
		self.dt=self.dt*self.tref
		self.time=self.time*self.tref

		# wake
		self.WzetaW=self.WzetaW*self.Uabs
		self.UzetaW=self.UzetaW*self.Uabs

		# time-stepping
		self.THgamma=self.THgamma/self.gref
		self.THgammaW=self.THgammaW/self.gref
		self.THGamma=self.THGamma/self.gref
		self.THGammaW=self.THGammaW/self.gref

		# time histories
		self.THZeta=self.THZeta*self.b 
		self.THZetaW=self.THZetaW*self.b
		self.THWzeta=self.THWzeta*self.Uabs
		self.THWzetaW=self.THWzetaW*self.Uabs
		self.THFaero_m=self.THFaero_m*self.Fref
		self.THFaero=self.THFaero*self.Fref
		self.THFdistr=self.THFdistr*self.Fref
		self.THVtot_zeta=self.THVtot_zeta*self.Uabs



	def build_flat_plate(self):
		'''
		Allocates variables for running a steady solution in time
		'''

		super().build_flat_plate()

		for tt in range(self.NT):
			self.THZeta[tt,:,:]=self.Zeta	



	def get_Cgamma_matrices(self):
		''' Produce matrices for propagation of wake - modified Ref.[4] '''

		M,Mw=self.M,self.Mw
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

		K,Kw=self.K,self.Kw
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

		M,K=self.M,self.K
		Iseg=np.zeros((K,M))
		Icoll=np.zeros((K,M))

		Iseg[:M,:]=np.eye(M)
		#Icoll
		wvcoll=np.zeros((K,))
		wvcoll[0]=1.-self.perc_interp 
		wvcoll[1]=self.perc_interp 
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

		# initialise
		Ndim=2
		M, Mw=self.M, self.Mw
		K, Kw=self.K, self.Kw
		NT=self.NT

		# Force 
		Fjouk=np.zeros((M,Ndim))  # at segments (grid in 2D)
		Fmass=np.zeros((self.M,Ndim)) # at collocation points

		### define constant matrices
		# convection wake intensity
		Cgamma,CgammaW=self.get_Cgamma_matrices()
		# convection wake coordinates
		Czeta,CzetaW=self.get_Czeta_matrices()
		# force interpolation
		Iseg,Icoll=self.get_force_matrices()

		# steady solution
		#self.Zeta=self.THZeta[0,:,:]
		self.THZetaW[0,:,:]=self.ZetaW
		self.dZetadt=0.0*self.Zeta
		if not(self._imp_start):
			self.solve_static_Gamma2d()
			# nondimensionalise
			self.nondimvars()
		else:
			# nondimensionalise
			self.nondimvars()
			self.build_AIC_Gamma2d()
			
		# LU factorisation
		if self._update_AIC is False:
			LU,P=scalg.lu_factor(self.A)

		# Time-stepping
		self.THgamma[0,:]=self.gamma
		self.THgammaW[0,:]=self.gammaW
		self.THGamma[0,:]=self.Gamma
		self.THGammaW[0,:]=self.GammaW
		self.THFaero[0,:]=self.FmatSta.sum(0)		

		if self.parallel==True: 
			pool = mpr.Pool(processes=self.PROCESSORS)
			self.Vind_zeta=0.0*self.Uzeta


		for tt in range(1,NT):

			### update aerofoil geometry
			# coordinates
			self.Zeta=self.THZeta[tt,:,:]
			# collocation points
			DZeta=np.diff(self.Zeta.T).T
			self.Cmat=self.Zeta[:K-1,:]+self.perc_coll*DZeta
			# normals
			kvers=np.array([0.,0.,1.])
			for ii in range(K-1):
				self.Nmat[ii,:]=\
				    np.array([-DZeta[ii,1],DZeta[ii,0]])/np.linalg.norm(
				    	                                            DZeta[ii,:])  

			### update wake geometry
			for dd in range(Ndim):
				self.ZetaW[:,dd]=np.dot(Czeta,self.Zeta[:,dd])+\
				                 np.dot(CzetaW,
				                 	    self.ZetaW[:,dd]+self.dt*
				                 	      (self.UzetaW[:,dd]+self.WzetaW[:,dd]))
			if self._savewake:
				self.THZetaW[tt,:,:]=self.ZetaW

			### update velocities
			self.WzetaW=np.dot(Czeta,self.Wzeta)+np.dot(CzetaW,self.WzetaW)
			self.Wzeta=self.THWzeta[tt,:,:]
			self.dZetadt=(self.Zeta-self.THZeta[tt-1,:,:])/self.dt

			### solve
			# velocity at collocation points
			self.get_Vcoll()
			self.Vcollperp=np.diag(np.dot(self.Nmat,self.Vcoll.T))

			# Propagate wake vorticity
			self.GammaW=np.dot(Cgamma,self.Gamma)+np.dot(CgammaW,self.GammaW)

			# build AIC matrices
			if self._update_AIC:
				self.build_AIC_Gamma2d()

			if self._quasi_steady:
				### enforce quasi-steady codition: 
				#	GammaW[0]=Gamma[-1]
				# or also:
				#	gammaW[0]=0
				# at each time-step
				self.Asys=self.A.copy()
				self.Asys[:,-1]+=self.AW[:,0]
				self.AW[:,0]=0.0
				# solve for bound circulation
				self.Gamma=np.linalg.solve(self.Asys,
					                -self.Vcollperp-np.dot(self.AW,self.GammaW))
				self.GammaW[0]=self.Gamma[-1]
			else:
				### returns 
				# 	GammaW[0](t+dt)=Gamma[-1](t)
				# or also
				# 	gammaW[0]=sum(gamma(t))-sum(gamma(t+dt))
				if self._update_AIC:
					self.Gamma=np.linalg.solve(self.A,
				                    -self.Vcollperp-np.dot(self.AW,self.GammaW))
				else:
					self.Gamma=scalg.lu_solve( (LU,P),
				                    -self.Vcollperp-np.dot(self.AW,self.GammaW))


			# Produce gamma
			self.gamma,self.gammaW=np.zeros((M,)),np.zeros((Mw,))
			self.gamma[0]=self.Gamma[0]
			for ii in range(1,M):
				self.gamma[ii]=self.Gamma[ii]-self.Gamma[ii-1]
			self.gammaW[0]=self.GammaW[0]-self.Gamma[-1]
			for ii in range(1,Mw):
				self.gammaW[ii]=self.GammaW[ii]-self.GammaW[ii-1]

			# induced velocity at grid points:
			if self.parallel==True:
				self.get_induced_velocity_parall(pool)			
			else:
				self.get_induced_velocity()

			# Total velocity
			self.Vtot_zeta=self.Uzeta+self.Wzeta+self.Vind_zeta-self.dZetadt

			### Force
			# static - Joukovski - at segments=first M grid points
			for nn in range(M):
				Fjouk[nn,:]=-self.gamma[nn]*\
				          np.array([-self.Vtot_zeta[nn,1],self.Vtot_zeta[nn,0]])
			# dynamic - added mass - collocation points
			# total velocity at collocaiton points (interpolate)
			# Wsub computed into get_coll()
			#self.Vtot_coll=np.dot(self.Wsub,self.Vtot_zeta)
			# added mass force
			DZeta=np.diff(self.Zeta.T).T

			for nn in range(M):
				Fmass[nn,:]=-np.linalg.norm(DZeta[nn,:])*\
				    (self.Gamma[nn]-self.THGamma[tt-1,nn])/self.dt*\
				                                                 self.Nmat[nn,:]
			
			# interpolate force over grid
			Faero_m=np.dot(Icoll,Fmass)
			self.Fmat=np.dot(Iseg,Fjouk)+Faero_m

			# store
			self.THVtot_zeta[tt,:,:]=self.Vtot_zeta
			self.THGamma[tt,:]=self.Gamma
			self.THGammaW[tt,:]=self.GammaW
			self.THgamma[tt,:]=self.gamma
			self.THgammaW[tt,:]=self.gammaW
			self.THFaero[tt,:]=self.Fmat.sum(0)
			self.THFaero_m[tt,:]=Faero_m.sum(0)
			self.THFdistr[tt,:,:]=self.Fmat


		# dimensionalise
		self.dimvars()

		# terminate
		self._exec_time=time.time()-start_time
		print('Done in %.1f sec!' % self._exec_time)

		if self.parallel==True:
			pool.close()
			pool.join() 




	def solve_dyn_Gamma2d_simple(self):
		'''
		This solution assumes that the aerofoil orientation/position does not 
		change in time. Only gust analysis and plunge motion are allowed.

		The wake is also unchanged.
		'''

		start_time = time.time()

		# initialise
		Ndim=2
		M, Mw=self.M, self.Mw
		K, Kw=self.K, self.Kw
		NT=self.NT

		# Force 
		Fjouk=np.zeros((M,Ndim))  # at segments (grid in 2D)
		Fmass=np.zeros((self.M,Ndim)) # at collocation points

		### define constant matrices
		# convection wake intensity
		Cgamma,CgammaW=self.get_Cgamma_matrices()
		# convection wake coordinates
		Czeta,CzetaW=self.get_Czeta_matrices()
		# force interpolation
		Iseg,Icoll=self.get_force_matrices()

		# steady solution
		#self.Zeta=self.THZeta[0,:,:]
		self.THZetaW[0,:,:]=self.ZetaW
		#self.dZetadt=0.0*self.Zeta
		if not(self._imp_start):
			self.solve_static_Gamma2d()
			# nondimensionalise
			self.nondimvars()
		else:
			# nondimensionalise
			self.nondimvars()
			self.build_AIC_Gamma2d()

		# LU factorisation
		LU,P=scalg.lu_factor(self.A)

		# Time-stepping
		self.THgamma[0,:]=self.gamma
		self.THgammaW[0,:]=self.gammaW
		self.THGamma[0,:]=self.Gamma
		self.THGammaW[0,:]=self.GammaW
		self.THFaero[0,:]=self.FmatSta.sum(0)		


		##### Things not required in time-loop

		# collocation points
		DZeta=np.diff(self.Zeta.T).T
		self.Cmat=self.Zeta[:K-1,:]+self.perc_coll*DZeta

		# normals
		kvers=np.array([0.,0.,1.])
		for ii in range(K-1):
			self.Nmat[ii,:]=\
			    np.array([-DZeta[ii,1],DZeta[ii,0]])/np.linalg.norm(DZeta[ii,:])

		for tt in range(1,NT):

			##### useless stuff kept for post-processing
			### update aerofoil geometry
			if self._savewake:
				self.THZetaW[tt,:,:]=self.ZetaW

			### update velocities
			self.Wzeta=self.THWzeta[tt,:,:]
			self.dZetadt=(self.THZeta[tt,:,:]-self.THZeta[tt-1,:,:])/self.dt

			### solve
			# velocity at collocation points
			self.get_Vcoll()
			self.Vcollperp=np.diag(np.dot(self.Nmat,self.Vcoll.T))

			# Propagate wake vorticity
			self.GammaW=np.dot(Cgamma,self.Gamma)+np.dot(CgammaW,self.GammaW)

			# solve for circulation
			self.Gamma=scalg.lu_solve( (LU,P),
				                    -self.Vcollperp-np.dot(self.AW,self.GammaW))

			# Produce gamma
			self.gamma,self.gammaW=np.zeros((M,)),np.zeros((Mw,))
			self.gamma[0]=self.Gamma[0]
			for ii in range(1,M):
				self.gamma[ii]=self.Gamma[ii]-self.Gamma[ii-1]
			self.gammaW[0]=self.GammaW[0]-self.Gamma[-1]
			for ii in range(1,Mw):
				self.gammaW[ii]=self.GammaW[ii]-self.GammaW[ii-1]

			# induced velocity at grid points:
			self.get_induced_velocity()

			# Total velocity
			self.Vtot_zeta=self.Uzeta+self.Wzeta-self.dZetadt+self.Vind_zeta
			self.THVtot_zeta[tt,:,:]=self.Vtot_zeta
			### Force
			# static - Joukovski - at segments=first M grid points
			for nn in range(M):
				Fjouk[nn,:]=-self.gamma[nn]*\
				          np.array([-self.Vtot_zeta[nn,1],self.Vtot_zeta[nn,0]])

			# dynamic - added mass - collocation points
			# total velocity at collocaiton points (interpolate)
			# Wsub computed into get_coll()
			#self.Vtot_coll=np.dot(self.Wsub,self.Vtot_zeta)
			# added mass force


			for nn in range(M):
				Fmass[nn,:]=-np.linalg.norm(DZeta[nn,:])*\
				    (self.Gamma[nn]-self.THGamma[tt-1,nn])/self.dt*\
				                                                 self.Nmat[nn,:]
			
			# interpolate force over grid
			Faero_m=np.dot(Icoll,Fmass)
			self.Fmat=np.dot(Iseg,Fjouk)+Faero_m

			# store
			self.THGamma[tt,:]=self.Gamma
			self.THGammaW[tt,:]=self.GammaW
			self.THgamma[tt,:]=self.gamma
			self.THgammaW[tt,:]=self.gammaW
			self.THFaero[tt,:]=self.Fmat.sum(0)
			self.THFaero_m[tt,:]=Faero_m.sum(0)
			self.THFdistr[tt,:,:]=self.Fmat


		# nondimensionalise
		self.dimvars()

		# terminate
		self._exec_time=time.time()-start_time
		print('Done in %.1f sec!' % self._exec_time)







# -----------------------------------------------------------------------------

if __name__=='__main__':


	# input


	# verify geometry



	Gamma_par=[]
	Mlist=[8,16,32]

	for M in Mlist:

		print('discretisation: M=%d, Mw=%d' %(M,20*M) )

		start_time = time.time()
		S=solver(T=.2,M=M,Mw=20*M,b=0.75,Uinf=np.array([20.,0.]), 
			                                      alpha=2.*np.pi/180.,rho=1.225)
		S.build_flat_plate()
		S.parallel=False
		S.solve_dyn_Gamma2d()
		print('\tSequential completed in %5.5f sec!' %(time.time()-start_time))
		Gamma_seq=S.Gamma.copy()


		start_time = time.time()
		S=solver(T=.2,M=M,Mw=20*M,b=0.75,Uinf=np.array([20.,0.]), 
			                                      alpha=2.*np.pi/180.,rho=1.225)
		S.parallel=True
		S.PROCESSORS=2
		S.build_flat_plate()
		S.solve_dyn_Gamma2d()
		print('\tParallel No.%.2d completed in %5.5f sec!' %(S.PROCESSORS,
			                                            time.time()-start_time))
		Gamma_par.append(S.Gamma.copy())



		start_time = time.time()
		S=solver(T=.2,M=M,Mw=20*M,b=0.75,Uinf=np.array([20.,0.]), 
			                                      alpha=2.*np.pi/180.,rho=1.225)
		S.parallel=True
		S.PROCESSORS=4
		S.build_flat_plate()
		S.solve_dyn_Gamma2d()
		print('\tParallel No.%.2d completed in %5.5f sec!' %(S.PROCESSORS,
			                                            time.time()-start_time))
		Gamma_par.append(S.Gamma.copy())


