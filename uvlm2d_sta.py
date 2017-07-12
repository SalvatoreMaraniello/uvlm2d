'''
2D UVLM solver
author: S. Maraniello
date: 7 Jun 2017
version: 1.0 "nondimensional solver"

Nomenclature as per "State space relatisation of p[otential flow unsteady 
aerodynamics with arbitrary kinematics

Ref.[1]: Anderson, Fundamentals of Aerodynamics
Ref.[2]: Simpson, Palacios and Maraniello, Scitech 2017
Ref.[3]: Katz and Plotkin, Low speed aerodynamics
'''

import scipy as sc
import scipy.linalg as scalg
import numpy as np
import multiprocessing as mpr
import matplotlib.pyplot as plt
from IPython import embed
import save


class solver():
	'''
	Steady state UVLM solver. 
	'''

	def __init__(self,M,Mw,b,Uinf,alpha,rho=1.225):

		# input
		self.M=M 
		self.Mw=Mw
		self.b=b
		self.alpha=alpha
		self.Uinf=Uinf
		self.Uabs=np.linalg.norm(Uinf)
		self.Udir=Uinf/self.Uabs
		self.rho=rho
		self.chord=2.*b
		self.qinf=0.5*rho*self.Uabs**2
		self.gref=self.b*self.Uabs
		self.Fref=self.rho*self.b*self.Uabs**2

		# vortex nodes
		self.K, self.Kw=M+1,Mw+1

		# parameters
		self.perc_ring=0.25  # backward shift of vortex panels wrt wing panels
		self.perc_coll=0.5   # collocation point shift wrt vortex ring TE
		self.perc_interp=0.5 # interpolation of velocity shift wrt vortex ring TE

		# initialise output
		self.gamma=np.zeros((M,))
		self.gammaW=np.zeros((Mw,))
		self.Gamma=np.zeros((M,))
		self.GammaW=np.zeros((Mw,))
		self.FmatSta = np.zeros((M,2))

		# settings
		self.PROCESSORS=4
		self.parallel=False#True


	def build_flat_plate(self):
		''' 
		Build geometry of a flat plate at an angle alpha w.r.t the global frame 
		Oxy. The collocation points are assumed at 50% of vortex ring.
		@warning: alpha is not necessarely the angle between plate and velocity;
		@warning: if the aerofoil has a velocity, the wake should be displaced. 
		This effect is not accounted here as in the static solution the wake has 
		no impact.
		'''

		# params
		Ndim=2
		K,Kw=self.K,self.Kw

		# grid coordinates
		self.Zeta=np.zeros((K,Ndim))
		self.ZetaW=np.zeros((Kw,Ndim))
		# wing panels coordinates
		self.Rmat=np.zeros((K,Ndim))
		# collocation points and normals
		self.Cmat=np.zeros((K-1,Ndim))
		self.Nmat=np.zeros((K-1,Ndim))

		# def wing panels coordinates
		rLE=2.*self.b*np.array([-np.cos(self.alpha),np.sin(self.alpha)])
		rTE=np.zeros((Ndim,))
		for ii in range(Ndim):
			self.Rmat[:,ii]=np.linspace(rLE[ii],rTE[ii],K)

		# def bound vortices "rings" coordinates
		dRmat=np.diff(self.Rmat.T).T
		self.Zeta[:K-1,:]=self.Rmat[:K-1,:]+self.perc_ring*dRmat
		self.Zeta[-1,:]=self.Zeta[-2,:]+dRmat[-1,:]

		# def wake vortices coordinates
		dl=2.*self.b/self.M
		twake=self.Uinf/np.linalg.norm(self.Uinf)
		EndWake=self.Zeta[-1,:]+self.Mw*dl*twake
		for ii in range(Ndim):
			self.ZetaW[:,ii]=np.linspace(self.Zeta[-1,ii],EndWake[ii],Kw)

		# def collocation points
		DZeta=np.diff(self.Zeta.T).T
		self.Cmat=self.Zeta[:K-1,:]+self.perc_coll*DZeta

		# def normals such that (zeta2-zeta1)^nv=kv
		kvers=np.array([0.,0.,1.])
		for ii in range(K-1):
			self.Nmat[ii,:]=\
			    np.array([-DZeta[ii,1],DZeta[ii,0]])/np.linalg.norm(DZeta[ii,:])  

		# set velocity fields
		self.dZetadt=np.zeros((self.K,2)) # aerofoil
		self.Wzeta=np.zeros((self.K,2))   # gust
		# def background velocity at vortex grid
		self.Uzeta = np.zeros((K,2))
		for nn in range(K): 
			self.Uzeta[nn,:]=self.Uinf

		return self



	def get_Vcoll(self):
		'''
		Retrieve the velocity induced at the collocation points by:
		- the aerofoil motion
		- the free stream
		'''
		K=self.K

		### velocity at collocation points
		#Vcollref=(1.-perc)*Vzeta[:K,:]+perc*Vzeta[1:,:]
		# Build interpolation tensor, eq.(1)
		wvsub=np.zeros((K,))
		wvsub[0]=1.-self.perc_interp 
		wvsub[1]=self.perc_interp  
		self.Wsub=(scalg.circulant( wvsub ).T)[:self.M,:]
		self.Vcoll = np.dot(self.Wsub,self.Uzeta+self.Wzeta-self.dZetadt)

		return self.Vcoll


	def build_AIC_gamma2d(self):
		'''
		Get aerodynamics influence coefficent matrices. These assume that the
		solution is wrt the total vorticity, gamma, at each Zeta coordinate (and
		not wrt the vorticity of each vortex ring).
		This approach leads to a system whose size depends on the number of 
		segments (or vetices, is equivalent) on the aerodynamic grid. For 2D
		cases, this does not increase the system size.
		@note: it is very hard to get this wrong
		'''

		self.A=np.zeros((self.M,self.M))
		self.AW=np.zeros((self.M,self.Mw))

		for ii in range(self.M):
			# bound
			for jj in range(self.M):
				self.A[ii,jj]=np.dot( self.Nmat[ii,:],
					 biot_savart_2d(self.Cmat[ii,:] ,self.Zeta[jj,:],gamma=1.0))
			# wake
			for jj in range(self.Mw):
				self.AW[ii,jj]=np.dot( self.Nmat[ii,:], 
					 biot_savart_2d(self.Cmat[ii,:],self.ZetaW[jj,:],gamma=1.0))

		return self


	def build_AIC_Gamma2d(self):
		'''
		Get aerodynamics influence coefficent matrices. These assume that the
		solution is wrt the current vorticity of each vortex ring).
		This approach leads to a system whose size is equal to the total number
		of panels on the aerodynamic grid. 

		@note: the contribution of the last segment is neglected, as it would
		create an umbalance such that the static solution would depend on the
		wake length.
		'''

		# allocate AIC matrices - positive contribution
		self.build_AIC_gamma2d()

		# "negative" contribution - bound
		self.A[:,:-1]+=-self.A[:,1:]
		# TE vortex from wake
		self.A[:,-1]+=-self.AW[:,0]
		# "negative" contribution - wake
		self.AW[:,:-1]+=-self.AW[:,1:]
		# neglect contribution of last segment of wake
		#for ii in range(self.M):
		#	self.AW[ii,-1]+= -np.dot( self.Nmat[ii,:],
		#			biot_savart_2d(self.Cmat[ii,:] ,self.ZetaW[-1,:],gamma=1.0))

		return self


	def nondimvars(self):
		'''
		Nondimensionalise variables of solver. Use the following reference 
		quantities:
		-length: self.b
		-velocities: self.Uabs
		@note: gamma* and Gamma* arrays do not need normalisation.
		'''

		# grid coordinates
		self.Zeta=self.Zeta/self.b
		self.ZetaW=self.ZetaW/self.b
		# wing panels coordinates
		self.Rmat=self.Rmat/self.b
		# collocation points and normals
		self.Cmat=self.Cmat/self.b
		# set velocity fields
		self.dZetadt=self.dZetadt/self.Uabs
		self.Wzeta=self.Wzeta/self.Uabs
		self.Uzeta=self.Uzeta/self.Uabs

		# nondimensionalise gamma* and Gamma* arrays (dynamic sol)
		self.gamma=self.gamma/self.gref
		self.gammaW=self.gammaW/self.gref
		self.Gamma=self.Gamma/self.gref
		self.GammaW=self.GammaW/self.gref

		# output
		self.FmatSta=self.FmatSta/self.Fref


	def dimvars(self):
		'''
		Dimensionalise variables of solver. Use the following reference 
		quantities:
		-length: self.b
		-velocities: self.Uabs
		'''

		# grid coordinates
		self.Zeta=self.Zeta*self.b
		self.ZetaW=self.ZetaW*self.b
		# wing panels coordinates
		self.Rmat=self.Rmat*self.b
		# collocation points and normals
		self.Cmat=self.Cmat*self.b
		# set velocity fields
		self.dZetadt=self.dZetadt*self.Uabs
		self.Wzeta=self.Wzeta*self.Uabs
		self.Uzeta=self.Uzeta*self.Uabs

		# dimensionalise gamma* and Gamma* arrays
		self.gamma=self.gamma*self.gref
		self.gammaW=self.gammaW*self.gref
		self.Gamma=self.Gamma*self.gref
		self.GammaW=self.GammaW*self.gref

		# output
		self.FmatSta=self.FmatSta*self.Fref


	def solve_static_gamma2d(self):
		'''
		Solves for the total vorticity at each vortex line, gamma.
		@warning: the method assumes that the geometry has already been built.
		'''
		Ndim=2

		K,Kw=self.K,self.Kw
		M,Mw=self.M,self.Mw

		# Nondimensionalise
		self.nondimvars()

		# get velocity at collocation points from:
		# - free stream
		# - aerofil motion
		self.get_Vcoll()
		self.Vcollperp=np.diag(np.dot(self.Nmat,self.Vcoll.T))

		# get AIC matrices
		self.build_AIC_gamma2d()	
		
		### solve:
		#gamma: vorticity at Zeta[ii,:] for ii=0:M (for Zeta[K,:] gamma=0)
		self.gamma=np.linalg.solve(self.A,-self.Vcollperp)

		# Produce Gamma
		self.Gamma[0]=self.gamma[0]
		for ii in range(1,M):
			self.Gamma[ii]=self.Gamma[ii-1]+self.gamma[ii]
		self.GammaW=self.Gamma[-1]*np.ones((Mw,))

		# induced velocity at grid points:
		self.get_induced_velocity()

		# Total velocity
		self.Vtot_zeta=self.Uzeta+self.Wzeta-self.dZetadt+self.Vind_zeta

		### Force - Joukovski
		for nn in range(M):
			self.FmatSta[nn,:]=-self.gamma[nn]*\
			              np.array([-self.Vtot_zeta[nn,1],self.Vtot_zeta[nn,0]])

		# dimensionalise
		self.dimvars()



	def solve_static_Gamma2d(self):
		'''
		Solves for the vorticity at each vortex line, Gamma.
		@warning: the method assumes that the geometry has already been built.
		'''
		Ndim=2

		K,Kw=self.K,self.Kw
		M,Mw=self.M,self.Mw

		# nondimensionalise
		self.nondimvars()


		# get velocity at collocation points from:
		# - free stream
		# - aerofil motion
		self.get_Vcoll()
		self.Vcollperp=np.diag(np.dot(self.Nmat,self.Vcoll.T))

		# get AIC matrices
		self.build_AIC_Gamma2d()	
		# enforce GammaW[:]=Gamma[-1]
		self.Asys=self.A.copy()
		self.Asys[:,-1]+=self.AW.sum(1)


		### solve:
		self.Gamma=np.linalg.solve(self.Asys,-self.Vcollperp)
		self.GammaW=self.Gamma[-1]*np.ones((Mw,))

		# Produce gamma
		self.gamma[0]=self.Gamma[0]
		for ii in range(1,M):
			self.gamma[ii]=self.Gamma[ii]-self.Gamma[ii-1]
		self.gammaW[0]=self.GammaW[0]-self.Gamma[-1]
		for ii in range(1,Mw):
			self.gammaW[ii]=self.GammaW[ii]-self.GammaW[ii-1]


		# induced velocity at grid points:
		#self.get_induced_velocity()
		if self.parallel==True:
			pool=mpr.Pool(processes=self.PROCESSORS)
			self.Vind_zeta=0.0*self.Uzeta
			self.get_induced_velocity_parall(pool)
			pool.close()
			pool.join() 
		else:
			self.get_induced_velocity()

		# Total velocity
		self.Vtot_zeta=self.Uzeta+self.Wzeta-self.dZetadt+self.Vind_zeta

		### Force - Joukovski
		for nn in range(M):
			self.FmatSta[nn,:]=-self.gamma[nn]*\
			              np.array([-self.Vtot_zeta[nn,1],self.Vtot_zeta[nn,0]])

		# dimensionalise
		self.dimvars()


	def get_induced_velocity(self):
		'''
		Computes induced velocity over the aerodynamic grid points except for 
		the last point, K - the grid is outside the aerofoil at this point. 

		The property that at the TE the gamma=0 is not exploited. To use it,
		the for loop over the wake could start from 1.
		'''
		M,Mw=self.M,self.Mw

		# - at TE gamma=0
		self.Vind_zeta=0.0*self.Uzeta
		for nn in range(M):
			#print('Comp. ind. velocity over node %.2d'%nn)
			# bound vortices "ahead"
			for kk in range(nn):
				#print('\tAdding contribution bound vortex-segment %.2d'%kk)
				self.Vind_zeta[nn,:]+=biot_savart_2d(self.Zeta[nn,:],
					                             self.Zeta[kk,:],self.gamma[kk])
			# bound vortices "behind"
			for kk in range(nn+1,M):
				#print('\tAdding contribution bound vortex-segment %.2d'%kk)
				self.Vind_zeta[nn,:]+=biot_savart_2d(self.Zeta[nn,:],
					                             self.Zeta[kk,:],self.gamma[kk])
			# wake vortices
			for kk in range(Mw):
				#print('\tAdding contribution wake vortex-segment %.2d'%kk)
				self.Vind_zeta[nn,:]+=biot_savart_2d(self.Zeta[nn,:],
					                           self.ZetaW[kk,:],self.gammaW[kk])
			### Add last segment of wake
			#self.Vind_zeta[nn,:]+=-biot_savart_2d(self.Zeta[nn,:],
			#	                               self.ZetaW[-1,:],self.GammaW[-1])



	def _get_induced_velocity_at_nn(self,nn,M,Mw,Zeta,ZetaW,gamma,gammaW):
		'''
		Computes induced velocity over the aerodynamic grid point self.Zeta[nn].
		
		This funciton is used to parallelise get_induced_velocity
		'''

		Vind_zeta_nn=0.0

		for kk in range(nn):
			#print('\tAdding contribution bound vortex-segment %.2d'%kk)
			Vind_zeta_nn+=biot_savart_2d(Zeta[nn,:],Zeta[kk,:],gamma[kk])
		# bound vortices "behind"
		for kk in range(nn+1,M):
			#print('\tAdding contribution bound vortex-segment %.2d'%kk)
			Vind_zeta_nn+=biot_savart_2d(Zeta[nn,:],Zeta[kk,:],gamma[kk])
		# wake vortices
		for kk in range(Mw):
			#print('\tAdding contribution wake vortex-segment %.2d'%kk)
			Vind_zeta_nn+=biot_savart_2d(Zeta[nn,:],ZetaW[kk,:],gammaW[kk])
		### Add last segment of wake
		#self.Vind_zeta[nn,:]+=-biot_savart_2d(self.Zeta[nn,:],
		#	                                   self.ZetaW[-1,:],self.GammaW[-1])

		return Vind_zeta_nn


	def get_induced_velocity_parall(self,pool):
		'''
		Computes induced velocity over the aerodynamic grid points except for 
		the last point, K - the grid is outside the aerofoil at this point. 

		The property that at the TE the gamma=0 is not exploited. To use it,
		the for loop over the wake could start from 1.
		'''

		# create pool of processes
		#pool = mpr.Pool(processes=self.PROCESSORS)  # @UndefinedVariable
		results=[]

		for nn in range(self.M):     
		    results.append( pool.apply_async(
		                         self._get_induced_velocity_at_nn,
		                         args=(nn,self.M,self.Mw,self.Zeta,self.ZetaW,
		                         	                   self.gamma,self.gammaW)))
		# retrieve results
		self.Vind_zeta[:-1,:]=np.array([p.get() for p in results])
		# - 1. close the pool (memory in workers goes to zero) 
		# - 2. exit the worker processes (processes are killed)
		#pool.close()
		#pool.join() 
            



	def analytical(self):
		'''
		Analytical solution for circulation along a flat plate. The solution 
		assumes that the airspeed is along the x axis.
		Ref. Anderson eq 4.24 and 4.29
		@warning: the sign convention in Ref. is the opposite as here (see 
		eq.4.16) and has been modified accordingly here.
		'''
		thvec=np.linspace(0.,np.pi,101)
		self.xvec_an=self.b*(1.0-np.cos(thvec)) - 2.*self.b
		self.dGammadtheta_an=-2.0*self.alpha*self.Uinf[0]*\
		                                        (1.+np.cos(thvec))/np.sin(thvec)
		self.Gamma_an=-2.0*self.b*self.alpha*self.Uinf[0]*(thvec+np.sin(thvec))
		self.GammaTot_an=-2.0*self.b*self.alpha*self.Uinf[0]*np.pi
		self.Lift_an=-self.rho*self.Uinf[0]*self.GammaTot_an



	def save(self,savedir,h5filename):
		save.h5file(savedir,h5filename,*(self,))




def biot_savart_2d(cv,zeta,gamma=1.0):
	'''
	Compute induced velocity over cv from an infinite filament parallel to
	the z axis and intersecting the Oxy plane at point cv. The vorticity 
	gamma is positive when anti-clockwise.
	Ref. Katz and Plotkin, eq.(2.70)
	'''

	drv=cv-zeta
	drabs=np.linalg.norm(drv)
	qv = 0.5*gamma/np.pi/drabs**2 * np.array([-drv[1],drv[0]])

	return qv



# -----------------------------------------------------------------------------

if __name__=='__main__':

	import time
	import pp_uvlm2d as pp

	# input
	chord=2.
	ainf=20.*np.pi/180.
	S=solver(M=20,Mw=20*20,b=0.5*chord,
		     Uinf=10.*np.array([np.cos(ainf),np.sin(ainf)]),
		     alpha=10.*np.pi/180.,
		     rho=1.225)

	# verify geometry
	S.build_flat_plate()
	fig,ax=pp.visualise_grid(S)
	plt.close()

	### verify biot-savart
	#gamma=1.0
	#qLE=biot_savart_2d( S.Rmat[0,:], S.Rmat[-1,:], gamma=1.0 )
	#qLEexp=-1./(2.*np.pi*chord)*gamma

	# solution using "gamma"
	S.solve_static_Gamma2d()
	S.analytical()
	fig=plt.figure('Total vorticity', figsize=[10.,6.0])
	ax=fig.add_subplot(111)
	ax.plot(S.xvec_an,S.Gamma_an,'r',label=r'$\Gamma$ (analytical)')
	ax.plot(S.Zeta[:-1,0],S.Gamma,'kx',label=r'$\Gamma$ numerical')
	ax.legend()
	# force distribution
	ax2,fig2=pp.force_distr(S)
	plt.show()
	plt.close('all')

	# Aero forces
	Ftot=S.FmatSta.sum(0)
	Mte=0.0
	for nn in range(S.M):
		Mte+=S.FmatSta[nn,1]*S.Zeta[nn,0]-S.FmatSta[nn,0]*S.Zeta[nn,1]
	rAC=np.zeros((2,))
	rAC[0]=np.interp(0.25*S.M,np.linspace(0,S.M,S.M+1),S.Rmat[:,0])
	rAC[1]=np.interp(0.25*S.M,np.linspace(0,S.M,S.M+1),S.Rmat[:,1])
	Mac=Mte - (Ftot[1]*rAC[0] - Ftot[0]*rAC[1])
	# AC calculation
	rLE=S.Rmat[0,:]
	etaLE = 1. - Mte / (-Ftot[0]*rLE[1]+Ftot[1]*rLE[0])
	# Aero coeff.s
	CF=Ftot/(2.*S.b*S.qinf)/S.alpha
	CM=Mac/(4.*S.b**2*S.qinf)/S.alpha


	1/0
	# Test induced velocity in parallel
	S.PROCESSORS=4
	print('discretisation: M=%d, Mw=%d' %(S.M,S.Mw) )
	start_time = time.time()
	S.get_induced_velocity()
	print('Sequential completed in %5.5f sec!' %(time.time()-start_time))
	Vind_seq=S.Vind_zeta.copy()

	pool = mpr.Pool(processes=S.PROCESSORS)  # @UndefinedVariable
	start_time = time.time()
	S.get_induced_velocity_parall(pool)
	print('Parallel completed in %5.5f sec!' %(time.time()-start_time))
	Vind_par=S.Vind_zeta.copy()
	pool.close()
	pool.join() 


