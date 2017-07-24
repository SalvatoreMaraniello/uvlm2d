'''
2D linearised UVLM solver
author: S. Maraniello
date: 7 Jun 2017
version: 1.0.* "towards vectorisation/linearisation"

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

import uvlm2d_sta
from uvlm2d_sta import biot_savart_2d
import libder


class solver():#uvlm2d_sta.solver):
	'''
	Linearised static solver.
	'''


	def __init__(self,S0):
		''' 
		The method takes an instance of the static UVLM 2D solver, S0. 
		Quantities have the same nomenclature as nonlinear solver, but have
		to be intended as deltas w.r.t. the reference configuration in S0
		'''

		Ndim=2
		self.S0=S0 # Reference state

		# create mapping for derivs computation
		self.S0.mapping()

		### initialise Delta terms
		M,Mw,K,Kw=S0.M,S0.Mw,S0.K,S0.Kw

		# State
		self.Gamma=np.zeros((M,))
		self.GammaW=np.zeros((Mw,))
		self.dGammadt=np.zeros((M,))

		# Input
		self.Zeta=np.zeros((K,Ndim))
		self.dZetadt=np.zeros((K,Ndim))
		self.Wzeta=np.zeros((K,Ndim))

		# Output
		self.FmatSta=np.zeros((M,Ndim))

		# Other: collocation points and normals
		self.Zeta_c=np.zeros((K-1,Ndim))
		self.Nmat=np.zeros((K-1,Ndim))


		# # Utilities: some incremental quantities for geometry modules
		# self.dalpha=0.0
		# self.dUinf=0.0


	# def build_flat_plate(self):
	# 	''' 
	# 	Build geometry/flow field of a flat plate at an angle alpha w.r.t the 
	# 	global frame  Oxy.  

	# 	@warning: alpha is not necessarely the angle between plate and velocity;
	# 	@warning: if the aerofoil has a velocity, the wake should be displaced. 
	# 	This effect is not accounted here as in the static solution the wake has 
	# 	no impact.
	# 	'''

	# 	S0=self.S0

	# 	self.dRmat,self.Zeta,self.ZetaW,self.Wzeta=\
	# 		geo.build_flat_plate(S0.b,self.dalpha,self.dUinf,
	# 			                   S0.M,S0.Mw,S0.K,S0.Kw,S0.perc_ring)

	# 	return self


	def solve_static_Gamma2d(self):

		Ndim=2
		K,Kw=self.S0.K,self.S0.Kw
		M,Mw=self.S0.M,self.S0.Mw	

		# pointer to self.S0
		S0=self.S0

		# normalise
		self.nondimvars()
		S0.nondimvars()

		##### Delta velocity at collocation point 

		### increment velocity (airspeed/aerofoil movement) contributions
		self.Vcoll=np.dot(S0._Wcv,self.Wzeta-self.dZetadt)
		self.Vcollperp=np.dot(S0._Wnc[0,:,:],self.Vcoll[:,0])+\
		                                  np.dot(S0._Wnc[1,:,:],self.Vcoll[:,1])

		### bound geometry changes
		# 3D component of ind. velocity at collocation points
		Vind_bound=np.dot(S0.AA,S0.Gamma).T 
		Vind_bound_wake=np.dot(S0.AAWW,S0.GammaW).T 
		# Derivative matrices
		dVind_dZeta,dVindW_dZeta=self.der_Wnc0AGamma_dZeta()
		dVind_dZeta=dVind_dZeta+self.der_WncV0_dZeta(Vind_bound)
		dVindW_dZeta=dVindW_dZeta+self.der_WncV0_dZeta(Vind_bound_wake)
		# Reference velocities contribution
		dVref_dZeta=self.der_WncV0_dZeta(S0.Uzeta+S0.Wzeta-S0.dZetadt)

		# Linearised contribution
		dRHS_dZeta=dVind_dZeta+dVindW_dZeta+dVref_dZeta
		self.dVind=np.dot(dRHS_dZeta[0,:,:],self.Zeta[:,0])+\
				   						np.dot(dRHS_dZeta[1,:,:],self.Zeta[:,1])

		# solve
		self.Gamma=np.linalg.solve(S0.Asys,-self.Vcollperp-self.dVind)

		# P.P
		self.gamma=np.dot(S0._TgG,self.Gamma)
		self.gammaW=np.dot(S0._TgG_w,self.GammaW)+np.dot(S0._EgG,self.Gamma)


		###################################### Compute Output - STILL NONLINEAR
		#
		# The variables here defined are TOTAL, i.e. due to incremental +
		# reference displacement/velocities

		# Total Vorticity
		self.GammaTot=S0.Gamma+self.Gamma
		self.GammaWTot=S0.GammaW+self.GammaW
		self.gammaTot=S0.gamma+self.gamma
		self.gammaWTot=S0.gammaW+self.gammaW

		# Total deformations (wake only changed at bound/wake interface)
		self.ZetaTot=S0.Zeta+self.Zeta
		self.ZetaWTot=S0.ZetaW.copy()
		self.ZetaWTot[S0.Map_bw[:,1],:]=self.ZetaTot[S0.Map_bw[:,0]] 

		# Total induced velocity
		self.get_total_induced_velocity()

		# Total velocity
		self.Vtot_zeta=S0.Vtot_zeta+self.Wzeta-self.dZetadt+self.VindTot_zeta

		# Force - Joukovski
		for nn in range(M):
			self.FmatSta[nn,:]=-self.gammaTot[nn]*\
			              np.array([-self.Vtot_zeta[nn,1],self.Vtot_zeta[nn,0]])

		################################## end Compute output - STILL NONLINEAR

		# dimensionalise
		self.dimvars()
		S0.dimvars()


	def get_total_induced_velocity(self):
		'''
		Computes the total induced velocity (reference + incremental) over the 
		aerodynamic grid points except for the last point, K - the grid is 
		outside the aerofoil at this point. 

		The property that at the TE the gamma=0 is not exploited. To use it,
		the for loop over the wake could start from 1.

		@warning: remove method once output linearisation is completed
		'''

		Ndim=2
		K,Kw=self.S0.K,self.S0.Kw
		M,Mw=self.S0.M,self.S0.Mw	

		# pointer to self.S0
		S0=self.S0

		### Total induced velocity
		# - at TE gamma=0
		self.VindTot_zeta=0.0*S0.Uzeta
		for nn in range(M):
			# bound vortices "ahead"
			for kk in range(nn):
				#print('\tAdding contribution bound vortex-segment %.2d'%kk)
				self.VindTot_zeta[nn,:]+=biot_savart_2d(self.ZetaTot[nn,:],
					                       self.ZetaTot[kk,:],self.gammaTot[kk])
			# bound vortices "behind"
			for kk in range(nn+1,M):
				#print('\tAdding contribution bound vortex-segment %.2d'%kk)
				self.VindTot_zeta[nn,:]+=biot_savart_2d(self.ZetaTot[nn,:],
					                       self.ZetaTot[kk,:],self.gammaTot[kk])
			# wake vortices
			for kk in range(Mw):
				#print('\tAdding contribution wake vortex-segment %.2d'%kk)
				self.VindTot_zeta[nn,:]+=biot_savart_2d(self.ZetaTot[nn,:],
					                     self.ZetaWTot[kk,:],self.gammaWTot[kk])
			### Add last segment of wake
			#self.Vind_zeta[nn,:]+=-biot_savart_2d(self.Zeta[nn,:],
			#	                               self.ZetaW[-1,:],self.GammaW[-1])



	def der_WncV0_dZeta(self,V0):
		'''
		Computes the derivative of the normal velocity, Wnc*V, w.r.t. self.Zeta.
		V is an array of velocities at the collocation points.
		'''

		Ndim=2
		K=self.S0.K
		M=self.S0.M

		derWncV=np.zeros((Ndim,M,K))

		# loop collocation points
		for mm in range(M):

			# identify local nodes
			map_cv=self.S0.Map_cv[mm,:]

			# extract coordinates
			ZetaLocal=self.S0.Zeta[map_cv,:]

			# partial derivative
			derLocal=libder.der_WncV_dZeta(
				zeta01_x=ZetaLocal[0,0],zeta02_x=ZetaLocal[1,0],
						zeta01_y=ZetaLocal[0,1],zeta02_y=ZetaLocal[1,1],
													V0_x=V0[mm,0],V0_y=V0[mm,1])

			# allocate partial derivatives
			derWncV[0,mm,map_cv[0]]=derLocal[0]
			derWncV[0,mm,map_cv[1]]=derLocal[1]
			derWncV[1,mm,map_cv[0]]=derLocal[2]
			derWncV[1,mm,map_cv[1]]=derLocal[3]

		return derWncV



	def der_Wnc0AGamma_dZeta(self):
		'''
		Computes the derivative 
		d( Wnc0 * A(zeta) Gamma0 )/d(zeta)
		'''

		Ndim=2
		K=self.S0.K
		M=self.S0.M
		Mw=self.S0.Mw

		Der=np.zeros((Ndim,M,K))
		DerW=np.zeros((Ndim,M,K))


		# loop collocation points
		for mm in range(M):

			# identify nodes connected to collocation point
			map_cv=self.S0.Map_cv[mm,:]
			# extract collocation point coordinates and normal
			zeta_c=self.S0.Zeta_c[mm,:]
			nc=self.S0.Nmat[mm,:]
			# extract vertices of ring with collocation point
			zeta_a=self.S0.Zeta[map_cv[0],:]
			zeta_b=self.S0.Zeta[map_cv[1],:]

			# loop through bound vortex rings
			for vv in range(M):

				# identify nodes of vortex ring
				map_here=self.S0.Map_cv[vv,:]
				# extract coordinates
				ZetaLocal=self.S0.Zeta[map_here,:]

				# allocate space for derivatives
				DerLocal=np.zeros((8,))

				### Derivative of total induced velocity of ring vv
				cf=0.5/np.pi
				DerLocal=libder.der_Wnc0AGamma_dZeta(
					zeta01=ZetaLocal[0,:],zeta02=ZetaLocal[1,:],
							  zetaA=zeta_a, zetaB=zeta_b,zetaC=zeta_c,
					                      nvec=nc,CF=cf,gamma=self.S0.Gamma[vv])
				
				# allocate partial derivatives w.r.t. vv ring
				Der[0,mm,map_here]=Der[0,mm,map_here]+DerLocal[0:2]
				Der[1,mm,map_here]=Der[1,mm,map_here]+DerLocal[2:4]

				# allocate partial derivatives w.r.t. mm ring
				Der[0,mm,map_cv]=Der[0,mm,map_cv]+DerLocal[4:6]
				Der[1,mm,map_cv]=Der[1,mm,map_cv]+DerLocal[6:8]


			# loop through wake vortex rings
			for vv in range(Mw):

				# identify last vortex (neglect last segment)
				if vv==Mw-1: allring=False
				else: allring=True

				# identify nodes of vortex ring
				map_here=self.S0.Map_cv_wake[vv,:]
				# extract coordinates
				ZetaLocal=self.S0.ZetaW[map_here,:]

				# allocate space for derivatives
				DerLocal=np.zeros((8,))

				### Derivative of total induced velocity of ring vv
				cf=0.5/np.pi
				DerLocal=libder.der_Wnc0AGamma_dZeta(
					zeta01=ZetaLocal[0,:],zeta02=ZetaLocal[1,:],
						zetaA=zeta_a, zetaB=zeta_b,zetaC=zeta_c,nvec=nc,
						         CF=cf,gamma=self.S0.GammaW[vv],allring=allring)

				# partial derivatives w.r.t. vv ring if at bound/wake interface
				for ii in range(2):
					kkvec=map_here[ii]==self.S0.Map_bw[:,1]
					if any(kkvec):
						pos_bound,pos_wake=self.S0.Map_bw[kkvec].reshape((2,))
						#print('added wake contrib. from node '\
						#	        '(bound: %d, wake: %d) to row %d'\
						#                               %(pos_bound,pos_wake,mm))
						DerW[:,mm,pos_bound]=\
						                DerW[:,mm,pos_bound]+DerLocal[[ii,2+ii]]

				# allocate partial derivatives w.r.t. mm ring
				DerW[0,mm,map_cv]=DerW[0,mm,map_cv]+DerLocal[4:6]
				DerW[1,mm,map_cv]=DerW[1,mm,map_cv]+DerLocal[6:8]


		return Der, DerW



	def nondimvars(self):
		'''
		Nondimensionalise variables of solver.
		@note: gamma* and Gamma* arrays do not need normalisation in static.
		'''

		# pointer to self.S0
		S0=self.S0

		# State
		self.Gamma=self.Gamma/S0.gref
		self.GammaW=self.GammaW/S0.gref
		self.dGammadt=self.dGammadt/(S0.gref/S0.tref)


		# Input
		self.Zeta=self.Zeta/S0.b
		self.dZetadt=self.dZetadt/S0.Uabs
		self.Wzeta=self.Wzeta/S0.Uabs

		# Output
		self.FmatSta=self.FmatSta/S0.Fref

		# Other: collocation points and normals
		self.Zeta_c=self.Zeta_c/S0.b



	def dimvars(self):
		'''
		Dimensionalise variables of solver.
		'''

		# pointer to self.S0
		S0=self.S0

		# State
		self.Gamma=self.Gamma*S0.gref
		self.GammaW=self.GammaW*S0.gref
		self.dGammadt=self.dGammadt*(S0.gref/S0.tref)


		# Input
		self.Zeta=self.Zeta*S0.b
		self.dZetadt=self.dZetadt*S0.Uabs
		self.Wzeta=self.Wzeta*S0.Uabs

		# Output
		self.FmatSta=self.FmatSta*S0.Fref

		# Other: collocation points and normals
		self.Zeta_c=self.Zeta_c*S0.b





if __name__=='__main__':

	import time
	import pp_uvlm2d as pp


	### build reference state
	Mw=10
	M=10
	alpha=2.*np.pi/180.
	ainf=2.0*np.pi/180.
	Uinf=20.*np.array([np.cos(ainf),np.sin(ainf)])
	chord=3.
	b=0.5*chord
	S0=uvlm2d_sta.solver(M,Mw,b,Uinf,alpha,rho=1.225)
	S0.build_camber_plate(Mcamb=10,Pcamb=4)
	S0.solve_static_Gamma2d()

	### check reference state
	S0.analytical()
	fig=plt.figure('Total vorticity', figsize=[10.,6.0])
	ax=fig.add_subplot(111)
	ax.plot(S0.xvec_an,S0.Gamma_an,'r',label=r'$\Gamma$ (analytical)')
	ax.plot(S0.Zeta[:-1,0],S0.Gamma,'kx',label=r'$\Gamma$ numerical')
	ax.legend()
	# force distribution
	ax2,fig2=pp.force_distr(S0)
	#plt.show()
	plt.close('all')

	### build linear solver
	Slin=solver(S0)

	# Perturb

	# Solve
	fact=1e-1
	Slin.Wzeta=Slin.S0.Uzeta*fact
	Slin.solve_static_Gamma2d()

	# Check force scaling
	LiftTotal=np.sum(Slin.FmatSta[:,1])
	Lift0=np.sum(Slin.S0.FmatSta[:,1])
	print('Lift factor: %f'%(LiftTotal/Lift0,)) 
	print('Expected: %f !'%(1.+fact)**2)

