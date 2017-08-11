'''
2D linearised UVLM solver
author: S. Maraniello
date: 15 Jul 2017

Nomenclature as per "State space relatisation of p[otential flow unsteady 
aerodynamics with arbitrary kinematics"

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
		self.Faero=np.zeros((K,Ndim))

		# Other: collocation points and normals
		self.Zeta_c=np.zeros((K-1,Ndim))
		self.Nmat=np.zeros((K-1,Ndim))

		# class name (for saving)
		self.name='solstalin'


	def solve_static_Gamma2d(self):

		Ndim=2
		K,Kw=self.S0.K,self.S0.Kw
		M,Mw=self.S0.M,self.S0.Mw	

		# pointer to self.S0
		S0=self.S0

		Fjouk=np.zeros((K,Ndim))

		# normalise
		self.nondimvars()
		S0.nondimvars()

		##### Delta velocity at collocation point 

		### increment velocity (airspeed/aerofoil movement) contributions
		self.Vcoll=np.dot(S0._Wcv,self.Wzeta-self.dZetadt)
		dVcollperp_a=np.dot(S0._Wnc[0,:,:],self.Vcoll[:,0])+\
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
		self.dVcoll_dZeta=dVind_dZeta+dVindW_dZeta+dVref_dZeta
		dVcollperp_b=np.dot(self.dVcoll_dZeta[0,:,:],self.Zeta[:,0])+\
				   			     np.dot(self.dVcoll_dZeta[1,:,:],self.Zeta[:,1])

		# solve
		self.Gamma=np.linalg.solve(S0.Asys,-dVcollperp_a-dVcollperp_b)
		self.GammaW[:]=self.Gamma[-1]

		# P.P
		self.gamma=np.dot(S0._TgG,self.Gamma)
		self.gammaW=np.dot(S0._TgG_w,self.GammaW)+np.dot(S0._EgG,self.Gamma)


		### Linearised delta force

		# Partial derivatived
		self.DFj_dGamma,self.DFj_dGammaW=self.der_Fjouk_dGamma_vind()
		self.DFj_dGamma=self.DFj_dGamma+self.der_Fjouk_dGamma_vtot0()
		self.DFj_dV=self.der_Fjouk_dV()
		self.DFj_dZeta=self.der_Fjouk_dZeta_vind()

		# incremental force
		for dd in range(Ndim):
			# state terms
			self.Faero[:,dd]=np.dot(self.DFj_dGamma[dd,:,:],self.Gamma)+\
								    np.dot(self.DFj_dGammaW[dd,:,:],self.GammaW)
			# input terms
			for tt in range(Ndim):
				self.Faero[:,dd]=self.Faero[:,dd]+\
					np.dot(self.DFj_dZeta[dd,:,tt,:],self.Zeta[:,tt])+\
								np.dot(self.DFj_dV[dd,:,tt,:],
						                    self.Wzeta[:,tt]-self.dZetadt[:,tt])

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
				
				try:
					# allocate partial derivatives w.r.t. vv ring
					Der[0,mm,map_here]=Der[0,mm,map_here]+DerLocal[0:2]
					Der[1,mm,map_here]=Der[1,mm,map_here]+DerLocal[2:4]

					# allocate partial derivatives w.r.t. mm ring
					Der[0,mm,map_cv]=Der[0,mm,map_cv]+DerLocal[4:6]
					Der[1,mm,map_cv]=Der[1,mm,map_cv]+DerLocal[6:8]
				except:
					embed()


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



	def der_Fjouk_dV(self):
		'''
		Computes the derivative of the Joukovski force (defined at each segment)
		w.r.t gust/aerofoil velocity at the same segments, 
		np.dot(Iseg,self.Wzeta) and np.dot(Iseg,self.dZetadt).
		'''

		Ndim=2
		K=self.S0.K
		M=self.S0.M
		Kcross2D=np.array([[0,-1],[1,0]])

		### derivative segment-to-segment
		# this assumes input velocity and output force are at the segments
		DerSeg=np.zeros((Ndim,M,Ndim,M))
		# loop through segments
		for ss in range(M):
			DerSeg[:,ss,:,ss]=-Kcross2D*self.S0.gamma[ss]

		### derivative grid to grid
		Der=np.zeros((Ndim,K,Ndim,K))

		#Interp. matrix from nodes to segment
		Iseg,Icoll=self.S0.get_force_matrices()
		for dd in range(Ndim):
			for nn in range(Ndim):
				Der[dd,:,nn,:]=np.dot( Iseg, np.dot(DerSeg[dd,:,nn,:],Iseg.T) )

		return Der



	def der_Fjouk_dGamma_vtot0(self):
		'''
		Computes the derivative of the Joukovski force (defined at each segment)
		w.r.t variation of bound circulations at constant total velocity.
		Note that this is not the only dependency of the Joukovski force w.r.t.
		the circulation, as an other contribution is given by the variation
		of induced velocity.
		'''

		Ndim=2
		K=self.S0.K
		M=self.S0.M
		Kcross2D=np.array([[0,-1],[1,0]])

		### induced velocity at grid points:
		#self.S0.get_induced_velocity()
		## Total velocity
		#self.S0.Vtot_zeta=self.Uzeta+self.Wzeta-self.dZetadt+self.Vind_zeta

		### derivative segment-to-segment
		# this assumes  output forces are at the segments
		DerSeg=np.zeros((Ndim,M,M))
		# loop through segments
		for ss in range(M):
			DerSeg[:,ss,ss]=-np.dot(Kcross2D,self.S0.Vtot_zeta[ss,:])

		### derivative grid to grid
		Der=np.zeros((Ndim,K,M))
		#Interp. matrix from nodes to segment
		Iseg,Icoll=self.S0.get_force_matrices()
		for dd in range(Ndim):
			Der[dd,:,:]=np.dot( Iseg, np.dot(DerSeg[dd,:,:],self.S0._TgG) )

		return Der


	def der_Fjouk_dGamma_vind(self):
		'''
		Computes the derivative of the Joukovski force (defined at each segment)
		w.r.t variation of circulations associated to changes of induced 
		velocity.
		Note that this is not the only dependency of the Joukovski force w.r.t.
		the circulation, as an other contribution is given by the variation
		of circulation at constant total velocity.
		'''

		Ndim=2
		K=self.S0.K
		M,Mw=self.S0.M,self.S0.Mw
		S0=self.S0

		Kcross2D=np.array([[0,-1],[1,0]])

		FFseg_gamma=np.zeros((Ndim,M,M))
		FFWWseg_gamma=np.zeros((Ndim,M,Mw))
		FFseg=np.zeros((Ndim,K,M))
		FFWWseg=np.zeros((Ndim,K,Mw))

		# loop bound segments
		for ii in range(M):
			# target segment
			zeta_seg=S0.Zeta[ii,:]

			# loop bound segments
			for jj in range(M):
				# neglect self-infuced velocity contribution
				if jj==ii:
					continue
				vind=biot_savart_2d(zeta_seg,S0.Zeta[jj,:],gamma=1.)
				# compute infl. coeff.
				FFseg_gamma[:,ii,jj]=-S0.gamma[ii]*np.dot(Kcross2D,vind)

			# loop wake segments
			for jj in range(Mw):
				# compute infl. coeff.
				vind=biot_savart_2d(zeta_seg,S0.ZetaW[jj,:],gamma=1.)
				FFWWseg_gamma[:,ii,jj]=-S0.gamma[ii]*np.dot(Kcross2D,vind)

		# convert
		Iseg,Icoll=self.S0.get_force_matrices()
		for dd in range(Ndim):
			FFseg[dd,:,:]=np.dot(Iseg,\
				                     np.dot(FFseg_gamma[dd,:,:],S0._TgG)+
                                          np.dot(FFWWseg_gamma[dd,:,:],S0._EgG))
			FFWWseg[dd,:,:]=np.dot(Iseg,np.dot(FFWWseg_gamma[dd,:,:],S0._TgG_w))

		return FFseg,FFWWseg



	def der_Fjouk_dZeta_vind(self):
		'''
		Derivative of Joukovski force w.r.t. changes of induced velocity at
		each segment associated to grid coordinates variations.
		'''

		Ndim=2
		K=self.S0.K
		M=self.S0.M
		Mw=self.S0.Mw

		Der=np.zeros((Ndim,M,Ndim,K))
		DerOut=np.zeros((Ndim,K,Ndim,K))

		# loop segments
		for ss in range(M):
			# warning: ss is both segment number and vertex number. In 3D these 
			# will be different

			# extract vertex coordinates
			zeta_c=self.S0.Zeta[ss,:]

			# loop through bound vortex rings
			for vv in range(M):

				# identify nodes of vortex ring
				map_here=self.S0.Map_cv[vv,:]
				# extract coordinates
				ZetaLocal=self.S0.Zeta[map_here,:]

				### Derivative of total induced velocity of ring vv
				cf=0.5/np.pi
				DerLocal=libder.der_Fjouk_ind_zeta(
					zeta01=ZetaLocal[0,:],zeta02=ZetaLocal[1,:],
						zetaCA=zeta_c,zetaCB=zeta_c,CF=cf,
						      Gamma=self.S0.Gamma[vv],gamma_s=self.S0.gamma[ss])
				
				# zero the terms associated to zeta01 and zetaCA if equal
				# (and similarly for zeta02 and zetaCB)
				for ii in range(len(map_here)):
					if map_here[ii]==ss:
						DerLocal[:,ii::2]=0.0 

				# allocate partial derivatives w.r.t. vv ring
				Der[:,ss,0,map_here]=Der[:,ss,0,map_here]+DerLocal[:,0:2]
				Der[:,ss,1,map_here]=Der[:,ss,1,map_here]+DerLocal[:,2:4]
				# allocate partial derivatives w.r.t. ss segment vertex
				Der[:,ss,0,ss]=Der[:,ss,0,ss]+DerLocal[:,4]+DerLocal[:,5]
				Der[:,ss,1,ss]=Der[:,ss,1,ss]+DerLocal[:,6]+DerLocal[:,7]


			# loop through wake vortex rings
			for vv in range(Mw):

				# identify last vortex (neglect last segment)
				if vv==Mw-1: allring=False
				else: allring=True

				# identify nodes of vortex ring
				map_here=self.S0.Map_cv_wake[vv,:]
				# extract coordinates
				ZetaLocal=self.S0.ZetaW[map_here,:]

				### Derivative of total induced velocity of ring vv
				cf=0.5/np.pi
				DerLocal=libder.der_Fjouk_ind_zeta(
					zeta01=ZetaLocal[0,:],zeta02=ZetaLocal[1,:],
					zetaCA=zeta_c,zetaCB=zeta_c,CF=cf,Gamma=self.S0.GammaW[vv],
					                  gamma_s=self.S0.gamma[ss],allring=allring)

				### zero the terms associated to zeta01,zeta02 if equal to zeta_c
				# this can occur only if were evaluating Fjouk on TE!
				# for ii in map_here:
				# 	if map_here[ii]==ss:
				# 		DerLocal[:,[ii,ii+2]]=0.0

				# Derivatives w.r.t. vv ring vertices are always zero except
				# when at the bound/wake interface. In 2D problems, when this
				# happens, zeta01 will have a non-zero contribution
				for ii in range(Ndim):
					kkvec=map_here[ii]==self.S0.Map_bw[:,1]
					if any(kkvec):
						pos_bound,pos_wake=self.S0.Map_bw[kkvec].reshape((2,))
						Der[:,ss,0,pos_bound]=Der[:,ss,0,pos_bound]+\
						                                          DerLocal[:,ii]
						Der[:,ss,1,pos_bound]=Der[:,ss,1,pos_bound]+\
						                                        DerLocal[:,ii+2]

				# allocate partial derivatives w.r.t. ss segment vertex
				Der[:,ss,0,ss]=Der[:,ss,0,ss]+DerLocal[:,4]+DerLocal[:,5]
				Der[:,ss,1,ss]=Der[:,ss,1,ss]+DerLocal[:,6]+DerLocal[:,7]

		# Project over lattice grid
		Iseg,Icoll=self.S0.get_force_matrices()
		for ii in range(Ndim):
			for jj in range(Ndim):
				DerOut[ii,:,jj,:]=np.dot(Iseg,Der[ii,:,jj,:]) 

		return DerOut




	def der_Fjouk_dZeta_vind_by_gamma(self):
		'''
		Derivative of Joukovski force w.r.t. changes of induced velocity at
		each segment associated to grid coordinates variations.
		'''

		Ndim=2
		K=self.S0.K
		M=self.S0.M
		Mw=self.S0.Mw

		Der=np.zeros((Ndim,M,Ndim,K))
		DerOut=np.zeros((Ndim,K,Ndim,K))


		# loop segments where force is computed
		for ss in range(M):
			# warning: ss is both segment number and vertex number. In 3D these 
			# will be different

			# extract vertex coordinates (segment where force is computed)
			zeta_c=self.S0.Zeta[ss,:]

			# loop through the segments producing velocity
			for tt in range(M):

				# neglect self-infuced velocity contribution
				if tt==ss:
					continue

				# position of segment producing velocity
				zeta01=self.S0.Zeta[tt,:]
				### Derivative of total induced velocity of ring tt
				cf=0.5/np.pi
				DerLocal=libder.der_Fjouk_ind_zeta_by_gamma(
					    zeta01=zeta01,zetaC=zeta_c,CF=cf,
							 gamma01=self.S0.gamma[tt],gammaC=self.S0.gamma[ss])
				# allocate partial derivatives w.r.t. tt seg.
				Der[:,ss,0,tt]=Der[:,ss,0,tt]+DerLocal[:,2]
				Der[:,ss,1,tt]=Der[:,ss,1,tt]+DerLocal[:,3]
				# allocate partial derivatives w.r.t. ss segment vertex
				Der[:,ss,0,ss]=Der[:,ss,0,ss]+DerLocal[:,0]
				Der[:,ss,1,ss]=Der[:,ss,1,ss]+DerLocal[:,1]


			# loop through wake vortex segment - last neglected as tt=0 is T.E.
			for tt in range(Mw):
				## no need to neglect tt=0 (T.E.) as force not computed
				#if tt==0: continue

				# position of segment producing velocity
				zeta01=self.S0.ZetaW[tt,:]

				### Derivative of total induced velocity of ring tt
				cf=0.5/np.pi
				DerLocal=libder.der_Fjouk_ind_zeta_by_gamma(
						zeta01=zeta01,zetaC=zeta_c,CF=cf,
					        gamma01=self.S0.gammaW[tt],gammaC=self.S0.gamma[ss])

				# Derivatives w.r.t. tt ring vertices are always zero except
				# when at the bound/wake interface (TE). In 2D problems, when 
				# this happens, zeta01 will have a non-zero contribution
				if tt in self.S0.Map_bw[:,1]:
					print('ss=%.2d tt=%.2d adding TE!' %(ss,tt))
					kk=np.where(tt==self.S0.Map_bw[:,1])[0][0]
					pos_bound=self.S0.Map_bw[kk,0]
					Der[:,ss,0,pos_bound]=Der[:,ss,0,pos_bound]+DerLocal[:,2]
					Der[:,ss,1,pos_bound]=Der[:,ss,1,pos_bound]+DerLocal[:,3]

				# allocate partial derivatives w.r.t. ss segment vertex
				Der[:,ss,0,ss]=Der[:,ss,0,ss]+DerLocal[:,0]
				Der[:,ss,1,ss]=Der[:,ss,1,ss]+DerLocal[:,1]

		# Project over lattice grid
		Iseg,Icoll=self.S0.get_force_matrices()
		for ii in range(Ndim):
			for jj in range(Ndim):
				DerOut[ii,:,jj,:]=np.dot(Iseg,Der[ii,:,jj,:]) 

		return DerOut



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
		self.Faero=self.Faero/S0.Fref

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
		self.Faero=self.Faero*S0.Fref

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

