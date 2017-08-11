'''
@author: salvatore maraniello
@date: 23 May 2017
@last update: 12 Jul 2017
@brief: tests for 2D UVLM dynamic solver
@note: test also implemented in python notebook

References:
[1] Simpson, R.J.S., Palacios, R. & Murua, J., 2013. Induced-Drag 
	Calculations in the Unsteady Vortex Lattice Method. AIAA Journal, 51(7), 
	pp.1775–1779.
'''
import os, sys
sys.path.append('..')

import numpy  as np
import matplotlib.pyplot as plt

import uvlm2d_dyn as uvlm
import pp_uvlm2d as pp
import analytical as an
import set_dyn, set_gust
import save

import warnings
import unittest
from IPython import embed



class Test_dyn(unittest.TestCase):
	'''
	Each method defined in this class contains a test case.
	@warning: by default, only functions whose name starts with 'test' is run.
	'''

	def setUp(self):
		''' Common piece of code to initialise the test '''

		self.SHOW_PLOT=False
		self.TOL_zero=1e-15		  # assess zero values      
		self.TOL_analytical=0.001 # assess accuracy of code
		self.TOL_numerical=1e-6   # assess changes between code versions
		pass


	def test_steady(self):
		'''
		Calculate steady case / very short wake can be used.
		'''

		# random geometry
		c=3.
		b=0.5*c
		uinf=20.0
		T=1.0
		WakeFact=2

		TimeList=[]
		THCFList=[]
		MList=[1,4,8]
		for mm in range(len(MList)):

		    M=MList[mm]
		    print('Testing steady aerofoil M=%d...' %M)
		    S=uvlm.solver(T=T,M=M,Mw=M*WakeFact,b=b,Uinf=np.array([uinf,0.]),
		                                          alpha=2.*np.pi/180.,rho=1.225)
		    S.build_flat_plate()
		    S.eps_Hall=1.0 # no correction
		    S.solve_dyn_Gamma2d()
		    TimeList.append(S.time)
		    THCFList.append(S.THFaero/S.qinf/S.chord)

		clist=['k','r','b','0.6',]
		fig = plt.figure('Aerodynamic forces',(12,4))
		ax=fig.add_subplot(131)
		for mm in range(len(MList)):
		    ax.plot(TimeList[mm],THCFList[mm][:,1],clist[mm],
		    	                                      label=r'M=%.2d'%MList[mm])
		ax.set_xlabel('time [s]')
		ax.set_title('Lift')
		ax.legend()
		ax=fig.add_subplot(132)
		for mm in range(len(MList)):
		    ax.plot(TimeList[mm],THCFList[mm][:,1]/S.alpha,clist[mm],
		    	                                      label=r'M=%.2d'%MList[mm])
		ax.set_xlabel('time [s]')
		ax.set_title('CL alpha')
		ax.legend()
		ax=fig.add_subplot(133)
		for mm in range(len(MList)):
		    ax.plot(TimeList[mm],THCFList[mm][:,0],clist[mm],
		    	                                      label=r'M=%.2d'%MList[mm])
		ax.set_xlabel('time [s]')
		ax.set_title('Drag')
		ax.legend()

		if self.SHOW_PLOT: plt.show()
		else: plt.close()

		### Testing
		maxDrag=0.0
		ClaExpected=2.*np.pi
		ClaNumericalV02=6.28190941
		ClaError=0.0
		ClaChangeNum=0.0
		for mm in range(len(MList)):
			# zero drag
			maxDragHere=np.max(np.abs(THCFList[mm][:,0]))
			if maxDragHere>maxDrag: maxDrag=maxDragHere
			# Cla relative error
			maxClaErrorHere=np.max(np.abs(
				                     THCFList[mm][:,1]/S.alpha/ClaExpected-1.0))
			if maxClaErrorHere>ClaError: ClaError=maxClaErrorHere
			# Cla change from previous releases
			maxClaChangeHere=np.max(np.abs(
				                 THCFList[mm][:,1]/S.alpha/ClaNumericalV02-1.0))
			if maxClaChangeHere>ClaChangeNum: ClaChangeNum=maxClaChangeHere


		self.assertTrue(maxDragHere<self.TOL_zero,msg=\
			'Max Drag %.2e above tolerance %.2e!' %(maxDragHere,self.TOL_zero))
		self.assertTrue(maxClaErrorHere<self.TOL_analytical,msg='Cla error %.2e'
			     ' above tolerance %.2e!'%(maxClaErrorHere,self.TOL_analytical))
		if maxClaChangeHere>self.TOL_numerical:
			warnings.warn('Relative change in Numerical Solution of %.2e !!!'
				                                              %maxClaChangeHere)


	def test_plunge(self):
		'''
		Plunge motion at low, medium-high and high reduced frequencies. 
		Especially at high reduced frequencies, the discretisation used are not
		refined enought to track correctly the lift. The accuracy of the induced
		drag is, instead, higher.
		Increasing the mesh (M) provides a big improvement, but the test case
		will be very slow.
		'''

		# random geometry/frequency
		c=3.
		b=0.5*c
		f0=2.#Hz
		w0=2.*np.pi*f0 #rad/s

		for case in ['debug','low','medium-high','high']:

			print('Testing aerofoil in plunge motion at %s frequency...' %case)

			if case is 'debug':
				ktarget=0.1
				H=0.02*b
				Ncycles=5.
				WakeFact=5
				M=4
			if case is 'low':
				ktarget=0.1
				H=0.2*b
				Ncycles=5.
				WakeFact=12
				M=8
			if case is 'medium-high':
				ktarget=.75
				H=0.2*b
				Ncycles=5.
				WakeFact=15
				M=20
			if case is 'high':
				ktarget=1.0
				H=0.02*b
				Ncycles=6.
				WakeFact=15
				M=20

			# speed/time
			uinf=b*w0/ktarget
			T=2.*np.pi*Ncycles/w0

			### solve
			S=uvlm.solver(T,M,Mw=M*WakeFact,b=b,Uinf=np.array([uinf,0.]),
				                                             alpha=0.,rho=1.225)
			S.build_flat_plate()
			S=set_dyn.plunge(S,f0=f0,H=H)
			S.eps_Hall=0.003
			S.solve_dyn_Gamma2d()

			### post-process
			hc_num=(S.THZeta[:,0,1]-H)/S.chord
			aeffv_num=np.zeros((S.NT))
			for tt in range(1,S.NT):
			    aeffv_num[tt]=-np.arctan(
			    	       (S.THZeta[tt,0,1]-S.THZeta[tt-1,0,1])/S.dt/S.Uinf[0])  
			THCF=S.THFaero/S.qinf/S.chord
			THCFmass=S.THFaero_m/S.qinf/S.chord
			THCFcirc=THCF-THCFmass        

			### Analytical solution
			hv_an=-H*np.cos(w0*S.time)
			hc_an=hv_an/S.chord
			dhv=w0*H*np.sin(w0*S.time)
			aeffv_an=np.arctan(-dhv/S.Uabs)
			# drag - Garrik
			Cdv=an.garrick_drag_plunge(w0,H,S.chord,S.rho,uinf,S.time)
			# lift - Theodorsen
			Ltot_an,Lcirc_an,Lmass_an=an.theo_lift(
				                             w0,0,H,S.chord,S.rho,S.Uinf[0],0.0)
			ph_tot=np.angle(Ltot_an)
			ph_circ=np.angle(Lcirc_an)
			ph_mass=np.angle(Lmass_an)
			CLtot_an=np.abs(Ltot_an)*np.cos(w0*S.time+ph_tot)/(S.chord*S.qinf)
			CLcirc_an=np.abs(Lcirc_an)*np.cos(w0*S.time+ph_circ)/(S.chord*S.qinf)
			CLmass_an=np.abs(Lmass_an)*np.cos(w0*S.time+ph_mass)/(S.chord*S.qinf)

			### Phase plots
			plt.close('all')
			fig = plt.figure('Induced drag in plunge motion -'
				                                  ' Phase vs kinematics',(10,6))
			ax=fig.add_subplot(111)
			ax.plot(180./np.pi*aeffv_an,Cdv,'k',lw=2,label=r'Analytical')
			ax.plot(180./np.pi*aeffv_num,THCF[:,0],'b',label=r'Numerical')
			ax.set_xlabel('deg')
			ax.set_title('M=%s'%M)
			ax.legend()

			fig = plt.figure('Lift in plunge motion - '
				                                   'Phase vs kinematics',(10,6))
			ax=fig.add_subplot(111)
			# analytical
			ax.plot(hc_an,CLtot_an,'k',lw=2,marker='o',markevery=(.3),
				                                                label=r'An Tot')
			# numerical
			ax.plot(hc_num,THCF[:,1],'b',label=r'Num Tot')
			ax.set_xlabel('h/c')
			ax.legend()

			### Time histories
			fig = plt.figure('Time histories',(12,5))
			ax=fig.add_subplot(121)
			ax.plot(S.time,hc_num,'y',lw=2,label='h/c')
			ax.plot(S.time,aeffv_num,'0.6',lw=2,label='Angle of attack [10 deg]')
			ax.plot(S.time,CLtot_an,'k',lw=2,label='An Tot')
			ax.plot(S.time,THCF[:,1],'b',lw=1,label='Num Tot')
			ax.set_xlabel('time')
			ax.set_xlim((1.-1./Ncycles)*T, T)
			ax.set_title('Lift coefficient')
			ax.legend()
			ax=fig.add_subplot(122)
			ax.plot(S.time, Cdv,'k',lw=2,label='An Tot')
			ax.plot(S.time, THCF[:,0],'b',lw=1,label='Num Tot')
			ax.set_xlim((1.-1./Ncycles)*T, T)
			ax.set_xlabel('time')
			ax.set_title('Drag coefficient')
			ax.legend()

			if self.SHOW_PLOT: plt.show()
			else: plt.close()

			embed()

			### Error in last cycle
			ttvec=S.time>float(Ncycles-1)/Ncycles*S.T
			ErCL=np.abs(THCF[ttvec,1]-CLtot_an[ttvec])/np.max(CLtot_an[ttvec])
			ErCD=np.abs(THCF[ttvec,0]-Cdv[ttvec])/np.max(np.abs(Cdv[ttvec]))
			# plt.plot(S.time[ttvec],ErCD,'b',label='CD')
			# plt.plot(S.time[ttvec],ErCL,'k',label='CL')
			# plt.legend()
			# plt.show()
			ErCLmax=np.max(ErCL)
			ErCDmax=np.max(ErCD)

			# Set tolerance and compare vs. expect results at low refinement
			if case is 'debug':
				tolCD,tolCL=2.5e3,1.5e3
				ErCLnumExp=1e5
				ErCDnumExp=1e5
			if case is 'low':
				tolCD,tolCL=2.5e-2,1.5e-2
				ErCLnumExp=0.011439946048750907
				ErCDnumExp=0.022575463702193873
			if case is 'medium-high':
				tolCD,tolCL=5.5e-2,8e-2
				ErCLnumExp=0.07797790766930307
				ErCDnumExp=0.051839455012024652
			if case is 'high':
				tolCD,tolCL=7e-2,10.5e-2
				ErCLnumExp=0.10392567505347514
				ErCDnumExp=0.068484800554194217

			# Check/raise error
			self.assertTrue(ErCDmax<tolCD,msg='Plunge case %s: relative CD '\
				       'error %.2e above tolerance %.2e!' %(case,ErCDmax,tolCD))
			self.assertTrue(ErCLmax<tolCL,msg='Plunge case %s: relative CL '\
				       'error %.2e above tolerance %.2e!' %(case,ErCLmax,tolCL))

			# Issue warnings
			changeCDnum=np.abs(ErCDmax-ErCDnumExp)
			changeCLnum=np.abs(ErCLmax-ErCLnumExp)
			if changeCDnum>self.TOL_numerical:
				warnings.warn('Relative change in CD numerical solution '\
					                                  'of %.2e !!!'%changeCDnum)
			if changeCLnum>self.TOL_numerical:
				warnings.warn('Relative change in CL numerical solution '\
					                                  'of %.2e !!!'%changeCLnum)

		#-----------------------------------------------------------------------

		# # Time histories of circulation
		# fig = plt.figure('Vortex rings circulation time history',(10,6))
		# ax=fig.add_subplot(111)
		# clist=['0.2','0.4','0.6']
		# Mlist=[0,int(S.M/2),S.M-1]
		# for kk in range(len(Mlist)):
		# 	mm=Mlist[kk]
		# 	ax.plot(S.time,S.THGamma[:,mm],color=clist[kk],label='M=%.2d'%(mm))
		# clist=['r','y','b']
		# MWlist=[0,int(S.Mw/2),S.Mw-1]
		# for kk in range(len(MWlist)):
		# 	mm=Mlist[kk]
		# 	ax.plot(S.time,S.THGammaW[:,mm],color=clist[kk],label='Mw=%.2d'%(mm))
		# ax.set_xlabel('time')
		# ax.set_ylabel('Gamma')
		# ax.legend()
		# plt.show()

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

		# Fcirc=S.THFaero-S.THFaero_m
		# fig = plt.figure('Lift contributions',(10,6))
		# ax=fig.add_subplot(111)
		# ax.plot(S.time,S.THFaero_m[:,1],'k',label=r'added mass')
		# ax.plot(S.time,Fcirc[:,1],'r',label=r'circulatory')
		# ax.set_xlabel('time [s]')
		# ax.set_ylabel('force [N]')
		# ax.legend()
		# plt.show()


	def test_sin_gust(self):

		'''
		Plunge motion at a fixed reduced frequency
		'''

		print('Testing aerofoil in sinusoidal gust...')

		# random geometry
		c=3.
		b=0.5*c

		# gust profile
		w0=0.1
		uinf=2.0
		L=10.0*c   # <--- gust wakelength

		# discretisation
		WakeFact=18
		Ncycles=10 # number of "cycles"
		Mfact=2
		if c>L: M=np.ceil(4*Mfact*c/L)
		else: M=Mfact*4

		# Numerical solution
		S=uvlm.solver(Ncycles*L/uinf,M,Mw=M*WakeFact,b=b,
			             Uinf=np.array([uinf,0.]),alpha=0.*np.pi/180.,rho=1.225)
		S.build_flat_plate()
		S=set_gust.sin(S,w0,L,ImpStart=False)
		S.eps_Hall=0.003
		S.solve_dyn_Gamma2d()
		THCF=S.THFaero/S.qinf/S.chord

		# Analytical solution
		CLv = an.sears_lift_sin_gust(w0,L,uinf,c,S.time)	

		fig = plt.figure('Aerodynamic force coefficients',(10,6))
		ax=fig.add_subplot(111)
		ax.set_title(r'Lift')
		ax.plot(S.time,CLv,'k',lw=2,label=r"Analytical")
		ax.plot(S.time,THCF[:,1],'b',label=r'Numerical')
		ax.legend()
		if self.SHOW_PLOT: plt.show()
		else: plt.close()

		### Detect peak and tim eof peak in last cycle
		# error on CL time-history not representative (amplified by lag)
		ttvec=S.time>float(Ncycles-1)/Ncycles*S.T
		# num sol.
		ttmax=np.argmax(THCF[ttvec,1])
		Tmax=S.time[ttvec][ttmax]
		CLmax=THCF[ttvec,1][ttmax] 
		# an. sol
		ttmax=np.argmax(CLv[ttvec])
		Tmax_an=S.time[ttvec][ttmax]
		CLmax_an=CLv[ttvec][ttmax] 
		# Expected values
		TmaxExp=139.875
		CLmaxExp=0.20297432667489032

		### Error
		ErCLmax=np.abs(CLmax/CLmax_an-1.0)
		Period=L/uinf
		ErTmax=np.abs((Tmax-Tmax_an)/Period)


		tolCL=3.5e-2
		tolTmax=6.e-2
		self.assertTrue(ErCLmax<tolCL,msg='Sinusoidal Gust: relative CLmax '\
				            'error %.2e above tolerance %.2e!' %(ErCLmax,tolCL))
		self.assertTrue(ErTmax<tolTmax,msg='Sinusoidal Gust: relative Tmax '\
				           'error %.2e above tolerance %.2e!' %(ErTmax,tolTmax))

		# Issue warnings
		changeTmaxnum=np.abs(Tmax-TmaxExp)
		changeCLnum=np.abs(CLmax-CLmaxExp)
		if changeCLnum>self.TOL_numerical:
			warnings.warn('Relative change in CLmax numerical solution '\
					                                  'of %.2e !!!'%changeCLnum)
		if changeTmaxnum>self.TOL_numerical:
			warnings.warn('Relative change in Tmax numerical solution '\
					                                'of %.2e !!!'%changeTmaxnum)

		# # visualise wake TH
		# fig=plt.figure('Wake TH', figsize=[10.,6.0])
		# ax=fig.add_subplot(121)
		# ax.set_title('Snapshots')
		# for ii in range(int(Ncycles/2)):
		# 	tt= 2*ii*4*Mfact
		# 	ax.plot(S.THZetaW[tt,:,0],S.THZetaW[tt,:,1],
		# 		                                      label=r'%.2e' %S.time[tt])
		# ax.set_xlabel('x [m]')
		# ax.set_xlabel('y [m]')
		# ax.legend()

		# ax=fig.add_subplot(122)
		# ax.set_title('TH specific points')
		# Npoints=5
		# for ii in range(0,S.Kw,int(S.Kw/(Npoints+1))):
		# 	ax.plot(S.time,S.THZetaW[:,ii,1],label=r'x=%.2e m' %S.ZetaW[ii,0] )
		# ax.set_xlabel('time [s]')
		# ax.set_xlabel('y [m]')
		# ax.legend()




	def test_impulse(self):
		'''
		Impulsive start (with/out Hall's correction) against Wagner solution
		'''

		print('Testing aerofoil impulsive start...')

		# set-up
		M=6
		WakeFact=20
		c=3.
		b=0.5*c
		uinf=20.0
		aeff=1.0*np.pi/180.
		T=8.0

		### Numerical solution - Hall's correction
		S=uvlm.solver(T=T,M=M,Mw=M*WakeFact,b=b,Uinf=np.array([uinf,0.]),
		                                                             alpha=aeff)
		S.build_flat_plate()
		S._imp_start=True
		S.eps_Hall=0.003
		S._update_AIC=True
		S.solve_dyn_Gamma2d()

		### Numerical solution - no Hall's correction
		S2=uvlm.solver(T=T,M=M,Mw=M*WakeFact,b=b,Uinf=np.array([uinf,0.]),
		                                                             alpha=aeff)
		S2.build_flat_plate()
		S2._imp_start=True
		S2.eps_Hall=1.0
		S2._update_AIC=True
		S2.solve_dyn_Gamma2d()

		### Analytical solution
		CLv_an=an.wagner_imp_start(aeff,uinf,c,S.time)

		##### Post-process numerical solution - Hall's correction
		THCF=S.THFaero/S.qinf/S.chord
		# Mass and circulatory contribution
		THCFmass=S.THFaero_m/S.qinf/S.chord
		THCFcirc=THCF-THCFmass

		##### Post-process numerical solution - no Hall's correction
		THCF2=S2.THFaero/S2.qinf/S2.chord
		# Mass and circulatory contribution
		THCFmass2=S2.THFaero_m/S2.qinf/S2.chord
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

		if self.SHOW_PLOT: plt.show()
		else: plt.close()

		# Final error
		ErCDend=np.abs(THCF[-1,0])
		ErCDend2=np.abs(THCF2[-1,0])
		ErCLend=np.abs(THCF[-1,1]-CLv_an[-1])/CLv_an[-1]
		ErCLend2=np.abs(THCF2[-1,1]-CLv_an[-1])/CLv_an[-1]

		# Expected values
		CLexp=0.10806840877933048
		CLexp2=0.10964318185934961
		changeCLnum=np.abs(THCF[-1,1]-CLexp)
		changeCLnum2=np.abs(THCF2[-1,1]-CLexp2)


		# Check/raise error
		self.assertTrue(ErCDend<self.TOL_analytical,msg=
			'Final CD %.2e (with Hall correction) above limit value of %.2e!'
			                                     %(ErCDend,self.TOL_analytical))
		self.assertTrue(ErCDend2<self.TOL_analytical,msg=
			'Final CD %.2e (without Hall correction) above limit value of %.2e!' 
			                                    %(ErCDend2,self.TOL_analytical))

		tolCL,tolCL2=1.5e-2,1.5e-3
		self.assertTrue(ErCLend<tolCL,msg='Relative CL error %.2e '\
			    '(with Hall correction) above tolerance %.2e!' %(ErCLend,tolCL))	
		self.assertTrue(ErCLend2<tolCL2,msg='Relative CL error %.2e '\
			'(without Hall correction) above tolerance %.2e!' %(ErCLend2,tolCL2))	

		if changeCLnum>self.TOL_numerical:
			warnings.warn('Relative change in CL numerical solution '\
				               '(with Hall correction) of %.2e !!!'%changeCLnum)
		if changeCLnum2>self.TOL_numerical:
			warnings.warn('Relative change in CL numerical solution '\
				           '(without Hall correction) of %.2e !!!'%changeCLnum2)



if __name__=='__main__':


	T=Test_dyn()
	T.setUp()
	#T.SHOW_PLOT=True
	
	### steady
	#T.test_steady()
	### plunge motion
	#T.test_plunge()
	### gust response
	#T.test_sin_gust()
	### wagner
	#T.test_impulse()

	### run all
	unittest.main()



# ------------------------------------------------------------------------------
# Discarded Test


		### Steady Plate with velocity
		# M=4
		# S=uvlm.solver(T=5.,M=M,Mw=M*WakeFact,b=b,Uinf=np.array([uinf,0.]),
		# 	      	                              alpha=0.*np.pi/180.,rho=1.225)
		# S.build_flat_plate()
		# aeff=3.*np.pi/180.
		# wplate=-np.tan(aeff)*uinf
		# S.dZetadt[:,1]=wplate
		# for tt in range(1,S.NT):
		# 	S.THZeta[tt,:,1]=wplate*S.time[tt]
		# S.eps_Hall=0.003
		# S.solve_dyn_Gamma2d()

		# qinf_eff=0.5*S.rho*np.linalg.norm([uinf,wplate])**2
		# THCF=S.THFaero/S.chord/qinf_eff
		# caeff,saeff=np.cos(aeff),np.sin(aeff)
		# Cmat=np.array([[caeff,saeff],[-saeff,caeff]])
		# for tt in range(S.NT):
		# 	THCF[tt,:]=np.dot(Cmat,THCF[tt,:])

		# fig = plt.figure('Aerodynamic forces in wind coord.',(10,6))
		# ax=fig.add_subplot(131)
		# for mm in range(len(MList)):
		# 	ax.plot(S.time,THCF[:,1],'k',label=r'M=%.2d'%MList[mm])
		# ax.set_xlabel('time [s]')
		# ax.set_title('Perpendicular')
		# #
		# ax=fig.add_subplot(132)
		# for mm in range(len(MList)):
		# 	ax.plot(S.time,THCF[:,1]/aeff,'k',label=r'M=%.2d'%MList[mm])
		# ax.set_xlabel('time [s]')
		# ax.set_title('Perpendicular - coefficient')
		# #
		# ax=fig.add_subplot(133)
		# for mm in range(len(MList)):
		# 	ax.plot(S.time,THCF[:,0],'k',label=r'M=%.2d'%MList[mm])
		# ax.set_xlabel('time [s]')
		# ax.set_title('Tangent')
		# plt.show()







