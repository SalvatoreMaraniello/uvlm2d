'''
@author: salvatore maraniello
@date: 23 May 2017
@brief: tests for 2D UVLM dynamic solver
@note: 

References:
[1] Simpson, R.J.S., Palacios, R. & Murua, J., 2013. Induced-Drag 
	Calculations in the Unsteady Vortex Lattice Method. AIAA Journal, 51(7), 
	pp.1775–1779.
'''


import os, sys
try: sys.path.append(os.environ["DIRuvlm2d"])
except: sys.path.append('../')

import numpy  as np
import matplotlib.pyplot as plt

import uvlm2d_dyn as uvlm
import pp_uvlm2d as pp
import analytical as an
import set_dyn, set_gust
import save

import unittest
from IPython import embed



class Test_dyn(unittest.TestCase):
	'''
	Each method defined in this class contains a test case.
	@warning: by default, only functions whose name starts with 'test' will be 
	run during testing.
	'''


	def setUp(self):
		''' Common piece of code to initialise the test '''
		pass


	def test_steady(self):
		'''
		Calculate steady case / very short wake can be used.
		'''


		##### Plate at an angle
		c=3.
		b=0.5*c
		uinf=20.0
		T=1.0
		MList=[1,4,8]
		WakeFact=2

		TimeList=[]
		THCFList=[]
		for mm in range(len(MList)):
			M=MList[mm]
			S=uvlm.solver(T=T,M=M,Mw=M*WakeFact,b=b,Uinf=np.array([uinf,0.]),
			      	      alpha=2.*np.pi/180.,rho=1.225)
			S.build_flat_plate()
			S.eps_Hall=0.003
			S.solve_dyn_Gamma2d()
			TimeList.append(S.time)
			THCFList.append(S.THFaero/S.qinf/S.chord)


		clist=['k','r','b','0.6',]
		fig = plt.figure('Aerodynamic forces',(10,6))
		#
		ax=fig.add_subplot(131)
		for mm in range(len(MList)):
			ax.plot(TimeList[mm],THCFList[mm][:,1],clist[mm],
				                                      label=r'M=%.2d'%MList[mm])
		ax.set_xlabel('time [s]')
		ax.set_title('Lift')
		ax.legend()
		#
		ax=fig.add_subplot(132)
		for mm in range(len(MList)):
			ax.plot(TimeList[mm],THCFList[mm][:,1]/S.alpha,clist[mm],
				                                      label=r'M=%.2d'%MList[mm])
		ax.set_xlabel('time [s]')
		ax.set_title('CL alpha')
		ax.legend()
		#
		ax=fig.add_subplot(133)
		for mm in range(len(MList)):
			ax.plot(TimeList[mm],THCFList[mm][:,0],clist[mm],
				                                      label=r'M=%.2d'%MList[mm])
		ax.set_xlabel('time [s]')
		ax.set_title('Drag')
		ax.legend()	
		#plt.show()


		##### Plate with velocity
		M=4
		S=uvlm.solver(T=5.,M=M,Mw=M*WakeFact,b=b,Uinf=np.array([uinf,0.]),
			      	                              alpha=0.*np.pi/180.,rho=1.225)
		S.build_flat_plate()
		aeff=3.*np.pi/180.
		wplate=-np.tan(aeff)*uinf
		S.dZetadt[:,1]=wplate
		for tt in range(1,S.NT):
			S.THZeta[tt,:,1]=wplate*S.time[tt]
		S.eps_Hall=0.003
		S.solve_dyn_Gamma2d()

		qinf_eff=0.5*S.rho*np.linalg.norm([uinf,wplate])**2
		THCF=S.THFaero/S.chord/qinf_eff
		caeff,saeff=np.cos(aeff),np.sin(aeff)
		Cmat=np.array([[caeff,saeff],[-saeff,caeff]])
		for tt in range(S.NT):
			THCF[tt,:]=np.dot(Cmat,THCF[tt,:])

		fig = plt.figure('Aerodynamic forces in wind coord.',(10,6))
		ax=fig.add_subplot(131)
		for mm in range(len(MList)):
			ax.plot(S.time,THCF[:,1],'k',label=r'M=%.2d'%MList[mm])
		ax.set_xlabel('time [s]')
		ax.set_title('Perpendicular')
		#
		ax=fig.add_subplot(132)
		for mm in range(len(MList)):
			ax.plot(S.time,THCF[:,1]/aeff,'k',label=r'M=%.2d'%MList[mm])
		ax.set_xlabel('time [s]')
		ax.set_title('Perpendicular - coefficient')
		#
		ax=fig.add_subplot(133)
		for mm in range(len(MList)):
			ax.plot(S.time,THCF[:,0],'k',label=r'M=%.2d'%MList[mm])
		ax.set_xlabel('time [s]')
		ax.set_title('Tangent')
		plt.show()


	def test_plunge(self,case=1,M=8,WakeFact=15):
		'''
		Plunge motion at a fixed reduced frequency
		'''

		### random geometry
		c=3.
		b=0.5*c

		### motion [Ref.[1]]
		if case==1:
			ktarget=.1
			H=0.2*b
			Ncycles=10.
		if case==2:
			ktarget=1.
			H=0.02*b
			Ncycles=16.
		if case==3:
			ktarget=.5
			H=0.05*b
			Ncycles=12.	
		if case==4:
			ktarget=.25
			H=0.2*b
			Ncycles=10.		
		if case==5:
			ktarget=.5
			H=0.2*b
			Ncycles=12.	
		if case==6:
			ktarget=.75
			H=0.2*b
			Ncycles=5

		f0=10.#Hz
		w0=2.*np.pi*f0 #rad/s
		uinf=b*w0/ktarget
		# Numerical solution
		T=2.*np.pi*Ncycles/w0

		###### Numerical solution
		S=uvlm.solver(T=T,M=M,Mw=M*WakeFact,b=b,Uinf=np.array([uinf,0.]),
		      	      alpha=0.*np.pi/180.,rho=1.225)
		S.build_flat_plate()
		S=set_dyn.plunge(S,f0=f0,H=H)
		S.eps_Hall=0.003
		S._update_AIC=True
		S._quasi_steady=False
		S.solve_dyn_Gamma2d()

		S.save(savedir='./res_plunge/k%.2d/'%(ktarget*10),
		       h5filename='M%.2dwk%.2dcyc%.2d'%(M,WakeFact,Ncycles))

		##### Analytical solution
		# Garrik - induced drag
		Cdv=an.garrick_drag_plunge(w0,H,S.chord,S.rho,uinf,S.time)
		hv=-H*np.cos(w0*S.time)
		dhv=w0*H*np.sin(w0*S.time)
		aeffv_an=np.arctan(-dhv/uinf)
		# Theodorsen - lift
		Ltot_an, Lcirc_an, Lmass_an = an.theo_lift(w0,A=0,H=H,c=S.chord,
									        rhoinf=S.rho,uinf=S.Uinf[0],x12=0.0)
		ph_tot,ph_circ,ph_mass=np.angle(Ltot_an),np.angle(Lcirc_an),np.angle(Lmass_an)
		hc_an=H/S.chord*np.cos(w0*S.time)
		CLtot_an=np.abs(Ltot_an)*np.cos(w0*S.time+ph_tot)/ (S.chord*S.qinf)
		CLcirc_an=np.abs(Lcirc_an)*np.cos(w0*S.time+ph_circ)/ (S.chord*S.qinf)
		CLmass_an=np.abs(Lmass_an)*np.cos(w0*S.time+ph_mass)/ (S.chord*S.qinf)

		##### Post-process numerical solution
		aeffv_num=aeffv_an # derivative is the same as analytical
		THCF=S.THFaero/S.qinf/S.chord
		# Mass and circulatory contribution
		THCFmass=np.zeros((S.NT,2))
		for tt in range(S.NT):
			THCFmass[tt,:]=S.THFmassC[tt,:,:].sum(0)/S.qinf/S.chord
		THCFcirc=THCF-THCFmass
		# Approximation of added mass
		THCFmass_approx=np.zeros((S.NT))
		Gtot_old=0.0
		for tt in range(1,S.NT):
			Gtot_new=S.THgamma[tt,:].sum()
			dgtot=Gtot_new-Gtot_old
			Gtot_old=Gtot_new
			THCFmass_approx[tt]=-S.rho*dgtot/S.dt/S.qinf


		##### Lift check - Numerical vs. Theodorsen
		fig = plt.figure('Lift coefficient',(10,6))
		ax=fig.add_subplot(111)
		### numerical
		ax.plot(S.time, THCF[:,1],'k',lw=1,label='Num Tot')
		#ax.plot(S.time, THCFmass[:,1],'b',lw=1,label='Num Mass')
		#ax.plot(S.time, THCFcirc[:,1],'r',lw=1,label='Num Circ')
		### approximation
		ax.plot(S.time, THCFmass_approx, 'y^', label='Approx Mass')
		# Theodorsen
		ax.plot(S.time, CLtot_an,'k--',lw=2,label='An Tot')
		#ax.plot(S.time, CLmass_an,'b--',lw=2,label='An Mass')
		#ax.plot(S.time, CLcirc_an,'r--',lw=2,label='An Circ')
		ax.set_xlabel('time')
		ax.set_ylabel('force')
		ax.set_title('Lift')
		ax.legend()


		fig = plt.figure('Drag coefficient',(10,6))
		ax=fig.add_subplot(111)
		ax.plot(S.time, THCF[:,0],'k',label='Num Tot')
		#ax.plot(S.time, THCFmass[:,0],'b',label='Num Mass')
		#ax.plot(S.time, THCFcirc[:,0],'r',label='Num Circ')
		ax.set_xlabel('time')
		ax.set_ylabel('force')
		ax.set_title('Drag')
		ax.legend()
		plt.show()

		fig = plt.figure('Induced drag in plunge motion - Phase vs kinematics',
			                                                             (10,6))
		ax=fig.add_subplot(111)
		ax.plot(180./np.pi*aeffv_an,Cdv,'k',label=r'Analytical')
		ax.plot(180./np.pi*aeffv_num,THCF[:,0],'b',label=r'Numerical')
		ax.set_xlabel('deg')
		ax.legend()
		fig.savefig('./figs/M%.2dW%.3dK%.2d_CD_F%.1dC%.1d_N%.2d_eps%.2e.png'
			                  %(M,WakeFact,10*ktarget,f0,c,Ncycles,S.eps_Hall) )


		# Time histories of circulation
		fig = plt.figure('Vortex rings circulation time history',(10,6))
		ax=fig.add_subplot(111)
		clist=['0.2','0.4','0.6']
		Mlist=[0,int(S.M/2),S.M-1]
		for kk in range(len(Mlist)):
			mm=Mlist[kk]
			ax.plot(S.time,S.THGamma[:,mm],color=clist[kk],label='M=%.2d'%(mm))
		clist=['r','y','b']
		MWlist=[0,int(S.Mw/2),S.Mw-1]
		for kk in range(len(MWlist)):
			mm=Mlist[kk]
			ax.plot(S.time,S.THGammaW[:,mm],color=clist[kk],label='Mw=%.2d'%(mm))
		ax.set_xlabel('time')
		ax.set_ylabel('Gamma')
		ax.legend()
		plt.show()


		# fig = plt.figure('Aero force at TE',(10,6))
		# ax=fig.add_subplot(111)
		# ax.plot(S.time,S.THFdistr[:,-1,0],'k',label=r'drag')
		# ax.plot(S.time,S.THFdistr[:,-1,1],'b',label=r'lift')
		# ax.set_xlabel('time')
		# ax.set_ylabel('force')
		# ax.legend()
		# plt.show()


		# ### plot net intensity of vortices along the aerofoil
		# for mm in range(S.M):
		# 	plt.plot(S.time,S.THgamma[:,mm],label='M=%.2d' %mm)
		# 	plt.xlabel('time [s]')
		# plt.legend(ncol=2)
		# plt.show()

		# # verify kutta condition
		# gte=np.zeros((S.NT))
		# for tt in range(1,S.NT):
		# 	Gtot_old=np.sum(S.THgamma[tt-1,:])
		# 	Gtot_new=np.sum(S.THgamma[tt,:])
		# 	gte[tt]=-(Gtot_new-Gtot_old)
		# fig = plt.figure('Net vorticity at TE',(10,6))
		# ax=fig.add_subplot(111)

		# ax.plot(S.time,S.THgammaW[:,0],color='k',label='Numerical')
		# ax.plot(S.time,gte,'r--',lw=2,label='Verification')

		# ax.set_xlabel('time')
		# ax.set_ylabel('Gamma')
		# ax.legend()
		# plt.show()



	def test_sin_gust(self):

		'''
		Plunge motion at a fixed reduced frequency
		'''

		# random geometry
		c=3.
		b=0.5*c

		# gust profile
		w0=0.003
		uinf=2.0
		L=10.0*c   # <--- gust wakelength

		# discretisation
		WakeFact=10
		Ncycles=10 # number of "cycles"
		Mfact=2
		if c>L: M=np.ceil(4*Mfact*c/L)
		else: M=Mfact*4

		# Numerical solution
		S=uvlm.solver(T=Ncycles*L/uinf,M=M,Mw=M*WakeFact,b=b,
			          Uinf=np.array([uinf,0.]),alpha=0.*np.pi/180.,rho=1.225)
		S.build_flat_plate()
		S=set_gust.sin(S,w0,L,ImpStart=False)
		S.eps_Hall=0.003
		S.solve_dyn_Gamma2d()

		# Analytical solution
		CLv = an.sears_lift_sin_gust(w0,L,uinf,c,S.time)		

		# Post-process
		THCF=S.THFaero/S.qinf/S.chord
		fig = plt.figure('Aerodynamic force coefficients',(10,6))
		ax=fig.add_subplot(111)
		ax.set_title(r'Lift')
		#ax.plot(S.time,THCF[:,0],'b',label=r'Drag')
		ax.plot(S.time,THCF[:,1],'k',label=r'UVLM')
		ax.plot(S.time,CLv,'r',label=r"Sear's")
		ax.legend()

		# visualise wake TH
		fig=plt.figure('Wake TH', figsize=[10.,6.0])
		ax=fig.add_subplot(121)
		ax.set_title('Snapshots')
		for ii in range(int(Ncycles/2)):
			tt= 2*ii*4*Mfact
			ax.plot(S.THZetaW[tt,:,0],S.THZetaW[tt,:,1],
				                                      label=r'%.2e' %S.time[tt])
		ax.set_xlabel('x [m]')
		ax.set_xlabel('y [m]')
		ax.legend()


		ax=fig.add_subplot(122)
		ax.set_title('TH specific points')
		Npoints=5
		for ii in range(0,S.Kw,int(S.Kw/(Npoints+1))):
			ax.plot(S.time,S.THZetaW[:,ii,1],label=r'x=%.2e m' %S.ZetaW[ii,0] )
		ax.set_xlabel('time [s]')
		ax.set_xlabel('y [m]')
		ax.legend()

		plt.show()



	def test_impulse(self,M=4,WakeFact=10):
		'''
		Impulsive start - Wagner solution
		'''

		### random geometry
		M=6
		WakeFact=20
		c=3.
		b=0.5*c
		uinf=20.0
		aeff=1.0*np.pi/180.
		T=5.0

		###### Numerical solution - Hall's correction
		S=uvlm.solver(T=T,M=M,Mw=M*WakeFact,b=b,Uinf=np.array([uinf,0.]),
		              alpha=aeff)
		S.build_flat_plate()
		S._imp_start=True
		S.eps_Hall=0.003
		S._update_AIC=True
		S.solve_dyn_Gamma2d()

		###### Numerical solution - no Hall's correction
		S2=uvlm.solver(T=T,M=M,Mw=M*WakeFact,b=b,Uinf=np.array([uinf,0.]),
		              alpha=aeff)
		S2.build_flat_plate()
		S2._imp_start=True
		S2.eps_Hall=1.0
		S2._update_AIC=True
		S2.solve_dyn_Gamma2d()

		##### Analytical solution
		CLv_an=an.wagner_imp_start(aeff,uinf,c,S.time)

		##### Post-process numerical solution - Hall's correction
		THCF=S.THFaero/S.qinf/S.chord
		# Mass and circulatory contribution
		THCFmass=np.zeros((S.NT,2))
		for tt in range(S.NT):
		    THCFmass[tt,:]=S.THFmassC[tt,:,:].sum(0)/S.qinf/S.chord
		THCFcirc=THCF-THCFmass

		##### Post-process numerical solution - no Hall's correction
		THCF2=S2.THFaero/S2.qinf/S2.chord
		# Mass and circulatory contribution
		THCFmass2=np.zeros((S2.NT,2))
		for tt in range(S2.NT):
		    THCFmass2[tt,:]=S2.THFmassC[tt,:,:].sum(0)/S2.qinf/S2.chord
		THCFcirc2=THCF2-THCFmass2



		plt.close('all')
		# non-dimensional time
		sv=2.0*S.Uabs*S.time/S.chord
		fig = plt.figure('Lift coefficient',(10,6))
		ax=fig.add_subplot(111)
		# Wagner
		ax.plot(0.5*sv, CLv_an,'0.6',lw=3,label='An Tot')
		# numerical
		ax.plot(0.5*sv, THCF[:,1],'k',lw=1,label='Num Tot - Hall')
		ax.plot(0.5*sv, THCFmass[:,1],'b',lw=1,label='Num Mass - Hall')
		ax.plot(0.5*sv, THCFcirc[:,1],'r',lw=1,label='Num Jouk - Hall')
		# numerical
		ax.plot(0.5*sv, THCF2[:,1],'k--',lw=2,label='Num Tot')
		ax.plot(0.5*sv, THCFmass2[:,1],'b--',lw=2,label='Num Mass')
		ax.plot(0.5*sv, THCFcirc2[:,1],'r--',lw=2,label='Num Jouk')
		ax.set_xlabel(r'$s/2=U_\infty t/c$')
		ax.set_ylabel('force')
		ax.set_title('Lift')
		ax.legend()

		fig2 = plt.figure('Lift coefficient - zoom',(10,6))
		ax=fig2.add_subplot(111)
		# Wagner
		ax.plot(0.5*sv, CLv_an,'0.6',lw=3,label='An Tot')
		# numerical
		ax.plot(0.5*sv, THCF[:,1],'k',lw=1,label='Num Tot - Hall')
		ax.plot(0.5*sv, THCFmass[:,1],'b',lw=1,label='Num Mass - Hall')
		ax.plot(0.5*sv, THCFcirc[:,1],'r',lw=1,label='Num Jouk - Hall')
		# numerical
		ax.plot(0.5*sv, THCF2[:,1],'k--',lw=2,label='Num Tot')
		ax.plot(0.5*sv, THCFmass2[:,1],'b--',lw=2,label='Num Mass')
		ax.plot(0.5*sv, THCFcirc2[:,1],'r--',lw=2,label='Num Jouk')
		ax.set_xlabel(r'$s/2=U_\infty t/c$')
		ax.set_ylabel('force')
		ax.set_title('Lift')
		ax.legend()
		ax.set_xlim(0.,6.)

		fig = plt.figure('Drag coefficient',(10,6))
		ax=fig.add_subplot(111)
		ax.plot(S.time, THCF[:,0],'k',label='Num Tot')
		ax.plot(S.time, THCFmass[:,0],'b',label='Num Mass')
		ax.plot(S.time, THCFcirc[:,0],'r',label='Num Jouk')
		ax.set_xlabel('time')
		ax.set_ylabel('force')
		ax.set_title('Drag')
		ax.legend()

		# Tiome histories of circulation
		fig = plt.figure('Vortex rings circulation time history',(10,6))
		ax=fig.add_subplot(111)
		clist=['0.2','0.4','0.6']
		Mlist=[0,int(S.M/2),S.M-1]
		for kk in range(len(Mlist)):
		    mm=Mlist[kk]
		    ax.plot(S.time,S.THGamma[:,mm],color=clist[kk],
		    	                                     label='M=%.2d - Hall'%(mm))
		clist=['r','y','b']
		MWlist=[0,int(S.Mw/2),S.Mw-1]
		for kk in range(len(MWlist)):
		    mm=Mlist[kk]
		    ax.plot(S.time,S.THGammaW[:,mm],color=clist[kk],
		    	                                    label='Mw=%.2d - Hall'%(mm))
		clist=['0.2','0.4','0.6']
		Mlist=[0,int(S.M/2),S.M-1]
		for kk in range(len(Mlist)):
		    mm=Mlist[kk]
		    ax.plot(S2.time,S2.THGamma[:,mm],color=clist[kk],linestyle='--',
		                                                    label='M=%.2d'%(mm))
		clist=['r','y','b']
		MWlist=[0,int(S.Mw/2),S.Mw-1]
		for kk in range(len(MWlist)):
		    mm=Mlist[kk]
		    ax.plot(S2.time,S2.THGammaW[:,mm],color=clist[kk],linestyle='--',
		                                                  label='Mw=%.2dl'%(mm))

		ax.set_xlabel('time')
		ax.set_ylabel('Gamma')
		ax.legend(ncol=2)
		plt.show()







if __name__=='__main__':


	T=Test_dyn()
	

	### steady
	T.test_steady()
	
	### plunge motion
	T.test_plunge(case=1,M=6,WakeFact=10)

	### gust response
	T.test_sin_gust()

	# wagner
	T.test_impulse(M=2,WakeFact=10)









