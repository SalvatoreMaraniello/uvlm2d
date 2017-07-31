'''
@author: salvatore maraniello
@date: 25 Jul 2017
@brief: tests for 2D linear UVLM dynamic solver
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

import uvlm2d_sta
import lin_uvlm2d_dyn
import pp_uvlm2d as pp
import analytical as an
import geo, set_dyn, set_gust
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

		self.SHOW_PLOT=True
		self.TOL_zero=1e-15		  # assess zero values      
		self.TOL_analytical=0.001 # assess accuracy of code
		self.TOL_numerical=1e-6   # assess changes between code versions
		pass


	def test_steady(self):
		'''
		Calculate steady case / very short wake can be used.
		'''

		### random geometry
		c=3.
		b=0.5*c
		uinf=20.0
		T=1.0
		WakeFact=2
		alpha0=1.00 * np.pi/180.
		dalpha=2.00 * np.pi/180.
		alpha_tot=alpha0+dalpha
		TimeList=[]
		THCFList=[]
		MList=[4,8,16]
		for mm in range(len(MList)):

			M=MList[mm]
			print('Testing steady aerofoil M=%d...' %M)
			### reference solution (static)
			S0=uvlm2d_sta.solver(M=M,Mw=M*WakeFact,b=b,
					 Uinf=np.array([uinf,0.]),alpha=alpha0,rho=1.225)
			S0.build_flat_plate()
			S0.eps_Hall=1.0 # no correction
			S0.solve_static_Gamma2d()
			### linearisation
			Slin=lin_uvlm2d_dyn.solver(S0,T)
			# perturb reference state
			ZetaRot=geo.rotate_aerofoil(S0.Zeta,dalpha)
			dZeta=ZetaRot-S0.Zeta
			Slin.Zeta=dZeta
			for tt in range(Slin.NT):
			    Slin.THZeta[tt,:,:]=dZeta
			# solve
			Slin.solve_dyn_Gamma2d()
			TimeList.append(Slin.time)
			THCFList.append(Slin.THFaero/S0.qinf/S0.chord)

		clist=['k','r','b','0.6',]
		fig = plt.figure('Aerodynamic forces',(12,4))
		ax=fig.add_subplot(131)
		for mm in range(len(MList)):
		    ax.plot(TimeList[mm],THCFList[mm][:,1],clist[mm],
		    	                                      label=r'M=%.2d'%MList[mm])
		ax.set_xlabel('time [s]')
		ax.set_title('Lift coefficient')
		ax.legend()
		ax=fig.add_subplot(132)
		for mm in range(len(MList)):
		    ax.plot(TimeList[mm],THCFList[mm][:,1]/alpha_tot,clist[mm],
		    	                                      label=r'M=%.2d'%MList[mm])
		ax.set_xlabel('time [s]')
		ax.set_title('CL alpha')
		ax.legend()
		ax=fig.add_subplot(133)
		for mm in range(len(MList)):
		    ax.plot(TimeList[mm],THCFList[mm][:,0],clist[mm],
		    	                                      label=r'M=%.2d'%MList[mm])
		ax.set_xlabel('time [s]')
		ax.set_title('Drag coefficient')
		ax.legend()

		if self.SHOW_PLOT: plt.show()
		else: plt.close()

		### Testing
		maxDrag=0.0
		ClaExpected=2.*np.pi
		ClaError=0.0
		ClaChangeNum=0.0


		for mm in range(len(MList)):
			# zero drag
			maxDragHere=np.max(np.abs(THCFList[mm][:,0]))
			if maxDragHere>maxDrag: maxDrag=maxDragHere
			# Cla relative error
			maxClaErrorHere=np.max(np.abs(
				                   THCFList[mm][:,1]/alpha_tot/ClaExpected-1.0))
			if maxClaErrorHere>ClaError: ClaError=maxClaErrorHere
			
		self.assertTrue(maxDragHere<self.TOL_zero,msg=\
			'Max Drag %.2e above tolerance %.2e!' %(maxDragHere,self.TOL_zero))
		self.assertTrue(maxClaErrorHere<self.TOL_analytical,msg='Cla error %.2e'
			     ' above tolerance %.2e!'%(maxClaErrorHere,self.TOL_analytical))




	def test_steady_incremental_speed(self):
		'''
		Calculate steady case / very short wake can be used. An incremental
		speed, which modified the effective angle of attack, is enforced on the
		linear model
		'''

		### random geometry
		c=3.
		b=0.5*c
		Uinf0=np.array([20.0,0.0])
		T=1.0
		WakeFact=2
		alpha0=1.00 * np.pi/180.
		dalpha=2.00 * np.pi/180.
		dalpha_inf=3.*np.pi/180.
		alpha_tot=alpha0+dalpha+dalpha_inf
		TimeList=[]
		THCFList=[]

		MList=[4,8,16]
		for mm in range(len(MList)):

			M=MList[mm]
			print('Testing steady aerofoil M=%d...' %M)
			### reference solution (static)
			S0=uvlm2d_sta.solver(M=M,Mw=M*WakeFact,b=b,
					                          Uinf=Uinf0,alpha=alpha0,rho=1.225)
			S0.build_flat_plate()
			S0.eps_Hall=1.0 # no correction
			S0.solve_static_Gamma2d()
			### linearisation
			Slin=lin_uvlm2d_dyn.solver(S0,T)
			# perturb reference state
			ZetaRot=geo.rotate_aerofoil(S0.Zeta,dalpha)
			dZeta=ZetaRot-S0.Zeta
			Slin.Zeta=dZeta
			Uinf=S0.Uabs*np.array([np.cos(dalpha_inf),np.sin(dalpha_inf)])
			#Slin.Wzeta[:,:]=(Uinf-Uinf0)
			Slin.dZetadt=-(Uinf-Uinf0)
			for tt in range(Slin.NT):
			    Slin.THZeta[tt,:,:]=dZeta
			    #Slin.THWzeta[tt,:,:]=(Uinf-Uinf0)
			    Slin.THdZetadt[tt,:,:]=-(Uinf-Uinf0)

			# solve
			Slin.solve_dyn_Gamma2d()
			TimeList.append(Slin.time)
			THCFList.append(Slin.THFaero/S0.qinf/S0.chord)


		# post-process: rotate force in wind axes
		THCFListRot=[]
		cainf,sainf=np.cos(dalpha_inf),np.sin(dalpha_inf)
		Rmat=np.array([[cainf,sainf],[-sainf,cainf]])
		for mm in range(len(MList)):
			THCFRot=1.0*THCFList[mm]
			for tt in range(THCFRot.shape[0]):
				THCFRot[tt,:]=np.dot(Rmat,THCFRot[tt,:])
			THCFListRot.append(THCFRot)


		clist=['k','r','b','0.6',]
		fig = plt.figure('Aerodynamic forces projected in wind axes',(12,4))
		ax=fig.add_subplot(131)
		for mm in range(len(MList)):
		    ax.plot(TimeList[mm],THCFListRot[mm][:,1],clist[mm],
		    	                                      label=r'M=%.2d'%MList[mm])
		ax.set_xlabel('time [s]')
		ax.set_title('Lift coefficient')
		ax.legend()
		ax=fig.add_subplot(132)
		for mm in range(len(MList)):
		    ax.plot(TimeList[mm],THCFListRot[mm][:,1]/alpha_tot,clist[mm],
		    	                                      label=r'M=%.2d'%MList[mm])
		ax.set_xlabel('time [s]')
		ax.set_title(r'$CL_\alpha$')
		ax.legend()
		ax=fig.add_subplot(133)
		for mm in range(len(MList)):
		    ax.plot(TimeList[mm],THCFListRot[mm][:,0],clist[mm],
		    	                                      label=r'M=%.2d'%MList[mm])
		ax.set_xlabel('time [s]')
		ax.set_title('Drag coefficient')
		ax.legend()

		if self.SHOW_PLOT: plt.show()
		else: plt.close()

		### Testing
		maxDrag=0.0
		ClaExpected=2.*np.pi
		ClaError=0.0
		ClaChangeNum=0.0


		for mm in range(len(MList)):
			# zero drag
			maxDragHere=np.max(np.abs(THCFListRot[mm][:,0]))
			if maxDragHere>maxDrag: maxDrag=maxDragHere
			# Cla relative error
			maxClaErrorHere=np.max(np.abs(
				                THCFListRot[mm][:,1]/alpha_tot/ClaExpected-1.0))
			if maxClaErrorHere>ClaError: ClaError=maxClaErrorHere

		embed()
			
		self.assertTrue(maxDragHere<self.TOL_zero,msg=\
			'Max Drag %.2e above tolerance %.2e!' %(maxDragHere,self.TOL_zero))
		self.assertTrue(maxClaErrorHere<self.TOL_analytical,msg='Cla error %.2e'
			     ' above tolerance %.2e!'%(maxClaErrorHere,self.TOL_analytical))




	def test_plunge(self):
		'''
		Plunge motion at low, medium-high and high reduced frequencies. 
		Especially at high reduced frequencies, the discretisation used are not
		refined enought to track correctly the lift. The accuracy of the induced
		drag is, instead, higher.
		Increasing the mesh (M) provides a big improvement, but the test case
		will be very slow.

		@warning: these test cases are different from those implemented in 
		test_dyn.py for the geometrically-exact solution.
		'''

		# random geometry/frequency
		c=3.
		b=0.5*c
		f0=2.#Hz
		w0=2.*np.pi*f0 #rad/s

		for case in ['debug']:#['debug','low','medium-high','high']:

			print('Testing aerofoil in plunge motion at %s frequency...' %case)

			if case is 'debug':
				ktarget=0.1
				H=0.01*b
				Ncycles=5.
				WakeFact=5
				M=4
			if case is 'low':
				ktarget=0.1
				H=0.01*b
				Ncycles=8#5.
				WakeFact=20#12
				M=80#12
			if case is 'medium-high':
				ktarget=.75
				H=0.01*b
				Ncycles=5.
				WakeFact=15
				M=28
			if case is 'high':
				ktarget=1.0
				H=0.01*b
				Ncycles=9#6.
				WakeFact=25#15
				M=120#30

			# speed/time
			uinf=b*w0/ktarget
			T=2.*np.pi*Ncycles/w0

			### reference static solution
			S0=uvlm2d_sta.solver(M=M,Mw=M*WakeFact,b=b,
			                       Uinf=np.array([uinf,0.]),alpha=0.0,rho=1.225)
			S0.build_flat_plate()
			S0.eps_Hall=1.0#0.003
			S0._saveout=False
			S0.solve_static_Gamma2d()

			# Linearised solution
			Slin=lin_uvlm2d_dyn.solver(S0,T)
			Slin=set_dyn.plunge(Slin,f0,H)
			Slin._savename='lindyn_plunge_%s_M%.3d_wk%.2d_ccl%.2d.h5'\
		                                              %(case,M,WakeFact,Ncycles)

			Slin.solve_dyn_Gamma2d()


			### post-process
			hc_num=(Slin.THZeta[:,0,1]-H)/S0.chord
			aeffv_num=np.zeros((Slin.NT))
			aeffv_num=-np.arctan(Slin.THdZetadt[:,0,1]/S0.Uinf[0])  
			#for tt in range(1,Slin.NT):
			#    aeffv_num[tt]=-np.arctan(
			#     (Slin.THZeta[tt,0,1]-Slin.THZeta[tt-1,0,1])/Slin.dt/S0.Uinf[0])  
			     
			THCF=Slin.THFaero/S0.qinf/S0.chord
			THCFmass=Slin.THFaero_m/S0.qinf/S0.chord
			THCFcirc=THCF-THCFmass        

			### Analytical solution
			hv_an=-H*np.cos(w0*Slin.time)
			hc_an=hv_an/S0.chord
			dhv=w0*H*np.sin(w0*Slin.time)
			aeffv_an=np.arctan(-dhv/S0.Uabs)
			# drag - Garrik
			Cdv=an.garrick_drag_plunge(w0,H,S0.chord,S0.rho,uinf,Slin.time)
			# lift - Theodorsen
			Ltot_an,Lcirc_an,Lmass_an=an.theo_lift(
				                          w0,0,H,S0.chord,S0.rho,S0.Uinf[0],0.0)
			ph_tot=np.angle(Ltot_an)
			ph_circ=np.angle(Lcirc_an)
			ph_mass=np.angle(Lmass_an)
			CLtot_an=np.abs(Ltot_an)*np.cos(w0*Slin.time+ph_tot)/(
				                                               S0.chord*S0.qinf)
			CLcirc_an=np.abs(Lcirc_an)*np.cos(w0*Slin.time+ph_circ)/(
				                                               S0.chord*S0.qinf)
			CLmass_an=np.abs(Lmass_an)*np.cos(w0*Slin.time+ph_mass)/(
				                                               S0.chord*S0.qinf)

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
			ax.plot(Slin.time,hc_num,'y',lw=2,label='h/c')
			ax.plot(Slin.time,aeffv_num,'0.6',lw=2,label='Angle of attack [10 deg]')
			ax.plot(Slin.time,CLtot_an,'k',lw=2,label='An Tot')
			ax.plot(Slin.time,THCF[:,1],'b',lw=1,label='Num Tot')
			ax.set_xlabel('time')
			ax.set_xlim((1.-1./Ncycles)*T, T)
			ax.set_title('Lift coefficient')
			ax.legend()
			ax=fig.add_subplot(122)
			ax.plot(Slin.time, Cdv,'k',lw=2,label='An Tot')
			ax.plot(Slin.time, THCF[:,0],'b',lw=1,label='Num Tot')
			ax.set_xlim((1.-1./Ncycles)*T, T)
			ax.set_xlabel('time')
			ax.set_title('Drag coefficient')
			ax.legend()

			if self.SHOW_PLOT: plt.show()
			else: plt.close()


			### Error in last cycle
			ttvec=Slin.time>float(Ncycles-1)/Ncycles*Slin.T
			ErCL=np.abs(THCF[ttvec,1]-CLtot_an[ttvec])/np.max(CLtot_an[ttvec])
			ErCD=np.abs(THCF[ttvec,0]-Cdv[ttvec])/np.max(np.abs(Cdv[ttvec]))
			# plt.plot(Slin.time[ttvec],ErCD,'b',label='CD')
			# plt.plot(Slin.time[ttvec],ErCL,'k',label='CL')
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

			warnings.warn('!!! Check for errors turned off !!!')
			# # Check/raise error
			# self.assertTrue(ErCDmax<tolCD,msg='Plunge case %s: relative CD '\
			# 	       'error %.2e above tolerance %.2e!' %(case,ErCDmax,tolCD))
			# self.assertTrue(ErCLmax<tolCL,msg='Plunge case %s: relative CL '\
			# 	       'error %.2e above tolerance %.2e!' %(case,ErCLmax,tolCL))

			# # Issue warnings
			# changeCDnum=np.abs(ErCDmax-ErCDnumExp)
			# changeCLnum=np.abs(ErCLmax-ErCLnumExp)
			# if changeCDnum>self.TOL_numerical:
			# 	warnings.warn('Relative change in CD numerical solution '\
			# 		                                  'of %.2e !!!'%changeCDnum)
			# if changeCLnum>self.TOL_numerical:
			# 	warnings.warn('Relative change in CL numerical solution '\
			# 		                                  'of %.2e !!!'%changeCLnum)

		#-----------------------------------------------------------------------

		# # Time histories of circulation
		# fig = plt.figure('Vortex rings circulation time history',(10,6))
		# ax=fig.add_subplot(111)
		# clist=['0.2','0.4','0.6']
		# Mlist=[0,int(S.M/2),S.M-1]
		# for kk in range(len(Mlist)):
		# 	mm=Mlist[kk]
		# 	ax.plot(Slin.time,S.THGamma[:,mm],color=clist[kk],label='M=%.2d'%(mm))
		# clist=['r','y','b']
		# MWlist=[0,int(S.Mw/2),S.Mw-1]
		# for kk in range(len(MWlist)):
		# 	mm=Mlist[kk]
		# 	ax.plot(Slin.time,S.THGammaW[:,mm],color=clist[kk],label='Mw=%.2d'%(mm))
		# ax.set_xlabel('time')
		# ax.set_ylabel('Gamma')
		# ax.legend()
		# plt.show()

		# fig = plt.figure('Aero force at TE',(10,6))
		# ax=fig.add_subplot(111)
		# ax.plot(Slin.time,S.THFdistr[:,-1,0],'k',label=r'drag')
		# ax.plot(Slin.time,S.THFdistr[:,-1,1],'b',label=r'lift')
		# ax.set_xlabel('time')
		# ax.set_ylabel('force')
		# ax.legend()
		# plt.show()

		# ### plot net intensity of vortices along the aerofoil
		# for mm in range(S.M):
		# 	plt.plot(Slin.time,S.THgamma[:,mm],label='M=%.2d' %mm)
		# 	plt.xlabel('time [s]')
		# plt.legend(ncol=2)
		# plt.show()

		# # verify kutta condition
		# gte=np.zeros((Slin.NT))
		# for tt in range(1,Slin.NT):
		# 	Gtot_old=np.sum(Slin.THgamma[tt-1,:])
		# 	Gtot_new=np.sum(Slin.THgamma[tt,:])
		# 	gte[tt]=-(Gtot_new-Gtot_old)
		# fig = plt.figure('Net vorticity at TE',(10,6))
		# ax=fig.add_subplot(111)

		# ax.plot(Slin.time,Slin.THgammaW[:,0],color='k',label='Numerical')
		# ax.plot(Slin.time,gte,'r--',lw=2,label='Verification')

		# ax.set_xlabel('time')
		# ax.set_ylabel('Gamma')
		# ax.legend()
		# plt.show()

		# Fcirc=Slin.THFaero-Slin.THFaero_m
		# fig = plt.figure('Lift contributions',(10,6))
		# ax=fig.add_subplot(111)
		# ax.plot(Slin.time,Slin.THFaero_m[:,1],'k',label=r'added mass')
		# ax.plot(Slin.time,Fcirc[:,1],'r',label=r'circulatory')
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
		WakeFact=25
		Ncycles=12 # number of "cycles"
		Mfact=25
		if c>L: M=np.ceil(4*Mfact*c/L)
		else: M=Mfact*4

		# Reference state
		S0=uvlm2d_sta.solver(M=M,Mw=M*WakeFact,b=b,
								   Uinf=np.array([uinf,0.]),alpha=0.0,rho=1.225)
		S0.build_flat_plate()
		S0._saveout=False
		S0.solve_static_Gamma2d()

		# Linearised model
		Slin=lin_uvlm2d_dyn.solver(S0,T=Ncycles*L/uinf)
		Slin=set_gust.sin(Slin,w0,L,ImpStart=False)
		Slin.eps_Hall=1.0#0.003
		Slin._savename='lindyn_sin_gust_M%.3d_wk%.2d_ccl%.2d.h5'\
		                                                   %(M,WakeFact,Ncycles)
		Slin.solve_dyn_Gamma2d()
		THCF=Slin.THFaero/S0.qinf/S0.chord

		# Analytical solution
		CLv = an.sears_lift_sin_gust(w0,L,uinf,c,Slin.time)	

		fig = plt.figure('Aerodynamic force coefficients',(10,6))
		ax=fig.add_subplot(111)
		ax.set_title(r'Lift')
		ax.plot(Slin.time,CLv,'k',lw=2,label=r"Analytical")
		ax.plot(Slin.time,THCF[:,1],'b',label=r'Numerical')
		ax.legend()
		if self.SHOW_PLOT: plt.show()
		else: plt.close()

		### Detect peak and tim eof peak in last cycle
		# error on CL time-history not representative (amplified by lag)
		ttvec=Slin.time>float(Ncycles-1)/Ncycles*Slin.T
		# num sol.
		ttmax=np.argmax(THCF[ttvec,1])
		Tmax=Slin.time[ttvec][ttmax]
		CLmax=THCF[ttvec,1][ttmax] 
		# an. sol
		ttmax=np.argmax(CLv[ttvec])
		Tmax_an=Slin.time[ttvec][ttmax]
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
		# 		                                      label=r'%.2e' %Slin.time[tt])
		# ax.set_xlabel('x [m]')
		# ax.set_xlabel('y [m]')
		# ax.legend()

		# ax=fig.add_subplot(122)
		# ax.set_title('TH specific points')
		# Npoints=5
		# for ii in range(0,S.Kw,int(S.Kw/(Npoints+1))):
		# 	ax.plot(Slin.time,S.THZetaW[:,ii,1],label=r'x=%.2e m' %S.ZetaW[ii,0] )
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


		### reference static solution - at zero - Hall's correction
		S0=uvlm2d_sta.solver(M=M,Mw=M*WakeFact,b=b,
		                       Uinf=np.array([uinf,0.]),alpha=0.0,rho=1.225)
		S0.build_flat_plate()
		S0.solve_static_Gamma2d()

		Slin1=lin_uvlm2d_dyn.solver(S0,T=T)
		ZetaRot=geo.rotate_aerofoil(S0.Zeta,aeff)
		dZeta=ZetaRot-S0.Zeta
		Slin1.Zeta=dZeta
		for tt in range(Slin1.NT):
			Slin1.THZeta[tt,:,:]=dZeta
		Slin1._imp_start=True
		Slin1.eps_Hall=0.003
		Slin1.solve_dyn_Gamma2d()



		### reference static solution - at zero - no Hall's correction
		S0=uvlm2d_sta.solver(M=M,Mw=M*WakeFact,b=b,
		                       Uinf=np.array([uinf,0.]),alpha=0.0,rho=1.225)
		S0.build_flat_plate()
		S0.eps_Hall=1.0
		S0.solve_static_Gamma2d()

		Slin2=lin_uvlm2d_dyn.solver(S0,T=T)
		Slin2.Zeta=dZeta
		for tt in range(Slin2.NT):
			Slin2.THZeta[tt,:,:]=dZeta
		Slin2._imp_start=True
		Slin2.eps_Hall=1.0
		Slin2.solve_dyn_Gamma2d()


		### Analytical solution
		CLv_an=an.wagner_imp_start(aeff,uinf,c,Slin1.time)

		##### Post-process numerical solution - Hall's correction
		THCF=Slin1.THFaero/S0.qinf/S0.chord
		# Mass and circulatory contribution
		THCFmass=Slin1.THFaero_m/S0.qinf/S0.chord
		THCFcirc=THCF-THCFmass

		##### Post-process numerical solution - no Hall's correction
		THCF2=Slin2.THFaero/S0.qinf/S0.chord
		# Mass and circulatory contribution
		THCFmass2=Slin2.THFaero_m/S0.qinf/S0.chord
		THCFcirc2=THCF2-THCFmass2

		plt.close('all')
		# non-dimensional time
		sv=2.0*S0.Uabs*Slin1.time/S0.chord

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

		fig3 = plt.figure("Lift coefficient - Hall's correction effect",(10,6))
		ax=fig3.add_subplot(111)
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
		ax.set_xlim(15.,0.5*sv[-1])
		ax.set_ylim( 0.10, 0.116 )


		fig = plt.figure('Drag coefficient',(10,6))
		ax=fig.add_subplot(111)
		ax.plot(Slin1.time, THCF[:,0],'k',label='Num Tot')
		ax.plot(Slin1.time, THCFmass[:,0],'b',label='Num Mass')
		ax.plot(Slin1.time, THCFcirc[:,0],'r',label='Num Jouk')
		ax.set_xlabel('time')
		ax.set_ylabel('force')
		ax.set_title('Drag')
		ax.legend()

		fig = plt.figure('Vortex rings circulation time history',(10,6))
		ax=fig.add_subplot(111)
		clist=['0.2','0.4','0.6']
		Mlist=[0,int(S0.M/2),S0.M-1]
		for kk in range(len(Mlist)):
		    mm=Mlist[kk]
		    ax.plot(Slin1.time,Slin1.THGamma[:,mm],color=clist[kk],
		    	                                     label='M=%.2d - Hall'%(mm))
		clist=['r','y','b']
		MWlist=[0,int(S0.Mw/2),S0.Mw-1]
		for kk in range(len(MWlist)):
		    mm=Mlist[kk]
		    ax.plot(Slin1.time,Slin1.THGammaW[:,mm],color=clist[kk],
		    	                                    label='Mw=%.2d - Hall'%(mm))
		clist=['0.2','0.4','0.6']
		Mlist=[0,int(S0.M/2),S0.M-1]
		for kk in range(len(Mlist)):
		    mm=Mlist[kk]
		    ax.plot(Slin2.time,Slin2.THGamma[:,mm],color=clist[kk],linestyle='--',
		                                                    label='M=%.2d'%(mm))
		clist=['r','y','b']
		MWlist=[0,int(S0.Mw/2),S0.Mw-1]
		for kk in range(len(MWlist)):
		    mm=Mlist[kk]
		    ax.plot(Slin2.time,Slin2.THGammaW[:,mm],color=clist[kk],linestyle='--',
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
			'(without Hall correction) above tolerance %.2e!'%(ErCLend2,tolCL2))	

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
	### steady incr. speed
	#T.test_steady_incremental_speed()

	### plunge motion
	T.test_plunge()
	### gust response
	#T.test_sin_gust()
	### wagner
	#T.test_impulse()

	### run all
	#unittest.main()

