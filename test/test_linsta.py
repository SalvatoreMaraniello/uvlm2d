'''
@author: salvatore maraniello
@date: 18 Jul 2017
@brief: tests for 2D linear UVLM static solver
@note: test also implemented in python notebook

References:
[1] Simpson, R.J.S., Palacios, R. & Murua, J., 2013. Induced-Drag 
	Calculations in the Unsteady Vortex Lattice Method. AIAA Journal, 51(7), 
	pp.1775–1779.
'''
import os, sys
sys.path.append('..')

import numpy  as np
import scipy.optimize as scopt
import matplotlib.pyplot as plt

import uvlm2d_sta as uvlm
import lin_uvlm2d_sta as linuvlm

import geo
import pp_uvlm2d as pp
import analytical as an
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


	def test_flat_rotation_about_zero(self):
		'''
		Calculate steady case / very short wake can be used. A flat plate at 
		at zero angle of attack is taken as reference condition. The state is
		perturbed of an angle dalpha. 
		'''

		MList=[1,8,16]
		DAlphaList=[0.1,0.5,1.,2.,4.,6.,8.,10.,12.,14.] # degs

		Nm=len(MList)
		Na=len(DAlphaList)

		CDlin,CLlin=np.zeros((Na,Nm)),np.zeros((Na,Nm))
		CDnnl,CLnnl=np.zeros((Na,Nm)),np.zeros((Na,Nm))


		for mm in range(len(MList)):

			# input: reference condition: flat plate at zero angle
			Mw=2
			M=MList[mm]
			alpha=0.*np.pi/180.
			chord=3.
			b=0.5*chord
			Uinf=np.array([20.,0.])
			rho=1.225
			S0=uvlm.solver(M,Mw,b,Uinf,alpha,rho=1.225)
			S0.build_flat_plate()
			S0.solve_static_Gamma2d()

			print('Testing steady aerofoil for M=%d...' %M)

			for aa in range(len(DAlphaList)):

				# Perturbation
				dalpha=np.pi/180.*DAlphaList[aa] # angle [rad]
				dUinf=0.0 # velocity [m/s]
				qinf_tot=0.5*rho*(S0.Uabs+dUinf)**2
				print('\talpha==%.2f deg' %DAlphaList[aa])


				### Linearised solution 

				# Perturb reference state
				ZetaRot=geo.rotate_aerofoil(S0.Zeta,dalpha)
				dZeta=ZetaRot-S0.Zeta
				Slin=linuvlm.solver(S0)
				Slin.Zeta=dZeta
				# solve
				Slin.solve_static_Gamma2d()
				# store data
				CDlin[aa,mm],CLlin[aa,mm]=np.sum(Slin.FmatSta,0)/\
										 (qinf_tot*Slin.S0.chord*(alpha+dalpha))

				### Reference nonlinear solution
				Sref=uvlm.solver(M,Mw,b,Uinf,alpha+dalpha,rho)
				Sref.build_flat_plate()
				# solve
				Sref.solve_static_Gamma2d()
				# store
				CDnnl[aa,mm],CLnnl[aa,mm]=np.sum(Sref.FmatSta,0)/\
										    (qinf_tot*Sref.chord*(alpha+dalpha))


		clist=['k','r','b','0.6',]
		fig = plt.figure('Aerodynamic coefficients',(12,4))
		ax1=fig.add_subplot(121)
		ax2=fig.add_subplot(122)

		for mm in range(Nm):
			#
			ax1.plot(DAlphaList,CDlin[:,mm],clist[mm],lw=Nm-mm,ls='--',
				                                  label=r'M=%.2d lin'%MList[mm])
			ax1.plot(DAlphaList,CDnnl[:,mm],clist[mm],lw=Nm-mm,
				                                  label=r'M=%.2d nnl'%MList[mm])
			#
			ax2.plot(DAlphaList,CLlin[:,mm],clist[mm],lw=Nm-mm,ls='--',
				                                  label=r'M=%.2d lin'%MList[mm])
			ax2.plot(DAlphaList,CLnnl[:,mm],clist[mm],lw=Nm-mm,
				                                  label=r'M=%.2d nnl'%MList[mm])

		ax1.set_xlabel(r'\alpha [deg]')
		ax1.set_title(r'CD')
		ax1.legend()
		ax2.set_xlabel(r'\alpha [deg]')
		ax2.set_title(r'CL')
		ax2.legend()

		if self.SHOW_PLOT: plt.show()
		else: plt.close()




	def test_flat_vert_disp_about_zero(self):
		'''
		Calculate steady case / very short wake can be used. A flat plate at 
		at zero angle of attack is taken as reference condition. The state is
		perturbed of an angle dalpha but the rotation is not exact (only 
		vertical displacements enforced)
		'''

		MList=[1,8,16]
		DAlphaList=[0.1,0.5,1.,2.,4.,6.,8.,10.,12.,14.] # degs

		Nm=len(MList)
		Na=len(DAlphaList)

		CDlin,CLlin=np.zeros((Na,Nm)),np.zeros((Na,Nm))
		CDnnl,CLnnl=np.zeros((Na,Nm)),np.zeros((Na,Nm))


		for mm in range(len(MList)):

			# input: reference condition: flat plate at zero angle
			Mw=2
			M=MList[mm]
			alpha=0.*np.pi/180.
			chord=3.
			b=0.5*chord
			Uinf=np.array([20.,0.])
			rho=1.225
			S0=uvlm.solver(M,Mw,b,Uinf,alpha,rho=1.225)
			S0.build_flat_plate()
			S0.solve_static_Gamma2d()

			print('Testing steady aerofoil for M=%d...' %M)

			for aa in range(len(DAlphaList)):

				# Perturbation
				dalpha=np.pi/180.*DAlphaList[aa] # angle [rad]
				dUinf=0.0 # velocity [m/s]
				qinf_tot=0.5*rho*(S0.Uabs+dUinf)**2
				print('\talpha==%.2f deg' %DAlphaList[aa])


				### Linearised solution 

				# Perturb reference state
				ZetaRot=geo.rotate_aerofoil(S0.Zeta,dalpha)
				dZeta=ZetaRot-S0.Zeta
				Slin=linuvlm.solver(S0)
				Slin.Zeta[:,1]=dZeta[:,1]
				# solve
				Slin.solve_static_Gamma2d()
				# store data
				CDlin[aa,mm],CLlin[aa,mm]=np.sum(Slin.FmatSta,0)/\
										 (qinf_tot*Slin.S0.chord*(alpha+dalpha))

				### Reference nonlinear solution
				Sref=uvlm.solver(M,Mw,b,Uinf,alpha+dalpha,rho)
				Sref.build_flat_plate()
				# solve
				Sref.solve_static_Gamma2d()
				# store
				CDnnl[aa,mm],CLnnl[aa,mm]=np.sum(Sref.FmatSta,0)/\
										    (qinf_tot*Sref.chord*(alpha+dalpha))


		clist=['k','r','b','0.6',]
		fig = plt.figure('Aerodynamic coefficients',(12,4))
		ax1=fig.add_subplot(121)
		ax2=fig.add_subplot(122)

		for mm in range(Nm):
			#
			ax1.plot(DAlphaList,CDlin[:,mm],clist[mm],lw=Nm-mm,ls='--',
				                                  label=r'M=%.2d lin'%MList[mm])
			ax1.plot(DAlphaList,CDnnl[:,mm],clist[mm],lw=Nm-mm,
				                                  label=r'M=%.2d nnl'%MList[mm])
			#
			ax2.plot(DAlphaList,CLlin[:,mm],clist[mm],lw=Nm-mm,ls='--',
				                                  label=r'M=%.2d lin'%MList[mm])
			ax2.plot(DAlphaList,CLnnl[:,mm],clist[mm],lw=Nm-mm,
				                                  label=r'M=%.2d nnl'%MList[mm])

		ax1.set_xlabel(r'\alpha [deg]')
		ax1.set_title(r'CD')
		ax1.legend()
		ax2.set_xlabel(r'\alpha [deg]')
		ax2.set_title(r'CL')
		ax2.legend()


		fig1 = plt.figure('Aerofoil',(12,4))
		ax1=fig1.add_subplot(111)
		ax1.plot(Sref.Zeta[:,0],Sref.Zeta[:,1],'k',lw=2,
		                                label=r'aerofoil 0 deg')
		ax1.plot(Sref.Zeta[:,0]+Slin.Zeta[:,0],Sref.Zeta[:,1]+Slin.Zeta[:,1],
			                 'b',lw=2,label=r'aerofoil %.f deg' %DAlphaList[-1])
		ax1.set_xlabel(r'x')
		ax1.set_title(r'y')
		ax1.legend()
		plt.show()

		if self.SHOW_PLOT: plt.show()
		else: plt.close()




	def test_camber_rotation(self):
		'''
		Calculate steady case / very short wake can be used. A cambered aerofoil 
		is tested at different angles of attack. Two linearisations, about zero
		and 10deg are used.
		'''

		DAlphaList=[0.0,0.1,0.2,0.3,0.5,1.,2.,4.,6.,8.,
		            9.5,9.7,9.8,9.9,10.,10.1,10.2,10.3,10.5,12.,14.,16.,18.] # degs

		MList=[20]
		Nm,mm=1,0
		Na=len(DAlphaList)

		CDlin01,CLlin01=np.zeros((Na,Nm)),np.zeros((Na,Nm))
		CDlin02,CLlin02=np.zeros((Na,Nm)),np.zeros((Na,Nm))
		CDnnl,CLnnl=np.zeros((Na,Nm)),np.zeros((Na,Nm))

		Llin01,Dlin01=np.zeros((Na,Nm)),np.zeros((Na,Nm))
		Llin02,Dlin02=np.zeros((Na,Nm)),np.zeros((Na,Nm))
		Lnnl,Dnnl=np.zeros((Na,Nm)),np.zeros((Na,Nm))

		# input:
		Mw=2
		M=MList[mm]
		chord=3.
		b=0.5*chord
		Uinf=np.array([20.,0.])
		rho=1.225

		# reference condition 1: aerofoil at 0 deg angle
		alpha01=0.*np.pi/180.
		S01=uvlm.solver(M,Mw,b,Uinf,alpha01,rho=1.225)
		S01.build_camber_plate(Mcamb=10,Pcamb=4)
		S01.solve_static_Gamma2d()
		Ftot0=np.sum(S01.FmatSta,0)

		# reference condition 2: aerofoil at 10 deg angle
		alpha02=10.*np.pi/180.
		S02=uvlm.solver(M,Mw,b,Uinf,alpha02,rho=1.225)
		S02.build_camber_plate(Mcamb=10,Pcamb=4)
		S02.solve_static_Gamma2d()


		print('Testing steady aerofoil for M=%d...' %M)

		for aa in range(len(DAlphaList)):

			# Perturbation
			alpha_tot=np.pi/180.*DAlphaList[aa]
			dUinf=0.0 # velocity [m/s]
			qinf_tot=0.5*rho*(S01.Uabs+dUinf)**2
			print('\talpha==%.2f deg' %DAlphaList[aa])


			### Linearised solution 01:
			# Perturb reference state
			dalpha01=alpha_tot-alpha01 # angle [rad]
			ZetaRot=geo.rotate_aerofoil(S01.Zeta,dalpha01)
			dZeta=ZetaRot-S01.Zeta
			Slin01=linuvlm.solver(S01)
			Slin01.Zeta=dZeta
			# solve
			Slin01.solve_static_Gamma2d()
			# store data
			Dlin01[aa,mm],Llin01[aa,mm]=np.sum(Slin01.FmatSta,0)
			dFtot01=np.sum(Slin01.FmatSta,0)-Ftot0
			CDlin01[aa,mm],CLlin01[aa,mm]=dFtot01/\
			                            (qinf_tot*Slin01.S0.chord*alpha_tot)


			### Linearised solution 02:
			# Perturb reference state
			dalpha02=alpha_tot-alpha02 # angle [rad]
			ZetaRot=geo.rotate_aerofoil(S02.Zeta,dalpha02)
			dZeta=ZetaRot-S02.Zeta
			Slin02=linuvlm.solver(S02)
			Slin02.Zeta=dZeta
			# solve
			Slin02.solve_static_Gamma2d()
			# store data
			Dlin02[aa,mm],Llin02[aa,mm]=np.sum(Slin02.FmatSta,0)
			dFtot02=np.sum(Slin02.FmatSta,0)-Ftot0
			CDlin02[aa,mm],CLlin02[aa,mm]=dFtot02/\
			                            (qinf_tot*Slin02.S0.chord*alpha_tot)


			### Reference nonlinear solution
			Sref=uvlm.solver(M,Mw,b,Uinf,alpha_tot,rho)
			Sref.build_camber_plate(Mcamb=10,Pcamb=4)
			# solve
			Sref.solve_static_Gamma2d()
			# store
			Dnnl[aa,mm],Lnnl[aa,mm]=np.sum(Sref.FmatSta,0)
			CDnnl[aa,mm],CLnnl[aa,mm]=np.sum(Sref.FmatSta,0)/\
			                                 (qinf_tot*Sref.chord*alpha_tot)


		clist=['k','r','b','0.6',]
		### Aerodynamic forces
		fig1 = plt.figure('Drag',(12,4))
		fig2 = plt.figure('Lift',(12,4))
		ax1=fig1.add_subplot(111)
		ax2=fig2.add_subplot(111)

		ax1.plot(DAlphaList,Dlin01[:,mm],'k',lw=2,ls='--',
		                                label=r'M=%.2d lin 0 deg'%MList[mm])
		ax1.plot(DAlphaList,Dlin02[:,mm],'b',lw=2,ls=':',
		                               label=r'M=%.2d lin 10 deg'%MList[mm])
		ax1.plot(DAlphaList,Dnnl[:,mm],'r',lw=Nm-mm,
		                                      label=r'M=%.2d nnl'%MList[mm])
		#
		ax2.plot(DAlphaList,Llin01[:,mm],'k',lw=2,ls='--',
		                                label=r'M=%.2d lin 0 deg'%MList[mm])
		ax2.plot(DAlphaList,Llin02[:,mm],'b',lw=Nm-mm,ls=':',
		                               label=r'M=%.2d lin 10 deg'%MList[mm])
		ax2.plot(DAlphaList,Lnnl[:,mm],'r',lw=2,
		                                      label=r'M=%.2d nnl'%MList[mm])

		ax1.set_xlabel(r'\alpha [deg]')
		ax1.set_title(r'Drag')
		ax1.legend()
		ax2.set_xlabel(r'\alpha [deg]')
		ax2.set_title(r'Lift')
		ax2.legend()
		plt.show()

		if self.SHOW_PLOT: plt.show()
		else: plt.close()




	def test_camber_external_speed(self):
		'''
		Calculate steady case / very short wake can be used. A cambered aerofoil,
		rotated of 0 and 10 deg w.r.t. the horizontal line, provides reference
		conditions for linearisation. An gust velocity is added to tilt the free 
		stream velocity and obtain effective angles of attack between 0,20 deg.
		'''

		DAlphaList=[0.0,0.1,0.2,0.3,0.5,1.,2.,4.,6.,8.,
		            9.5,9.7,9.8,9.9,10.,10.1,10.2,10.3,10.5,12.,14.,16.,18.] # degs
		#DAlphaList=[5.0]

		MList=[20]
		Nm,mm=1,0
		Na=len(DAlphaList)

		CDlin01,CLlin01=np.zeros((Na,Nm)),np.zeros((Na,Nm))
		CDlin02,CLlin02=np.zeros((Na,Nm)),np.zeros((Na,Nm))
		CDnnl,CLnnl=np.zeros((Na,Nm)),np.zeros((Na,Nm))

		Llin01,Dlin01=np.zeros((Na,Nm)),np.zeros((Na,Nm))
		Llin02,Dlin02=np.zeros((Na,Nm)),np.zeros((Na,Nm))
		Lnnl,Dnnl=np.zeros((Na,Nm)),np.zeros((Na,Nm))

		# input:
		Mw=2
		M=MList[mm]
		chord=3.
		b=0.5*chord
		Uinf=np.array([20.,0.])
		rho=1.225

		# reference condition 1: aerofoil at 0 deg angle
		alpha01=0.*np.pi/180.
		S01=uvlm.solver(M,Mw,b,Uinf,alpha01,rho=1.225)
		S01.build_camber_plate(Mcamb=10,Pcamb=4)
		S01.solve_static_Gamma2d()
		Ftot0=np.sum(S01.FmatSta,0)

		# reference condition 2: aerofoil at 10 deg angle
		alpha02=10.*np.pi/180.
		S02=uvlm.solver(M,Mw,b,Uinf,alpha02,rho=1.225)
		S02.build_camber_plate(Mcamb=10,Pcamb=4)
		S02.solve_static_Gamma2d()


		print('Testing steady aerofoil for M=%d...' %M)


		def get_wvec(aeff,a0,ux,uz):

			# get free stream angle
			a=aeff-a0
			tga=np.tan(a)

			### build residual
			def Fres(x):
				'''@warning: uses function variables!'''
				wx,wz=x
				F=np.zeros((2,))
				F[0]=tga*(ux+wx)-(uz+wz)
				F[1]=wx**2+2.*wx*ux+wz**2+2.*uz*wz
				return F

			# solve nnl equations
			x0=np.zeros((2,))
			wvec=scopt.newton_krylov(Fres,x0)

			asol=np.tan((uz+wvec[1])/(ux+wvec[0]))
			assert (asol-(aeff-a0))**2<1e-3, 'Solution not found!'

			return wvec


		for aa in range(len(DAlphaList)):

			# Perturbation
			alpha_eff=np.pi/180.*DAlphaList[aa]
			qinf_tot=0.5*rho*(S01.Uabs)**2
			print('\tAlpha effective=%.2f deg' %DAlphaList[aa])

			### Linearised solution 01:
			# Perturb reference state
			Slin01=linuvlm.solver(S01)
			wvec=get_wvec(alpha_eff,alpha01,Uinf[0],Uinf[1])
			Slin01.Wzeta[:,:]=wvec
			# solve
			Slin01.solve_static_Gamma2d()
			# store data
			Dlin01[aa,mm],Llin01[aa,mm]=np.sum(Slin01.FmatSta,0)
			dFtot01=np.sum(Slin01.FmatSta,0)-Ftot0
			CDlin01[aa,mm],CLlin01[aa,mm]=dFtot01/\
			                                (qinf_tot*Slin01.S0.chord*alpha_eff)

			### Linearised solution 02:
			# Perturb reference state
			Slin02=linuvlm.solver(S02)
			wvec=get_wvec(alpha_eff,alpha02,Uinf[0],Uinf[1])
			Slin02.Wzeta[:,:]=wvec
			# solve
			Slin02.solve_static_Gamma2d()
			# store data
			Dlin02[aa,mm],Llin02[aa,mm]=np.sum(Slin02.FmatSta,0)
			dFtot02=np.sum(Slin02.FmatSta,0)-Ftot0
			CDlin02[aa,mm],CLlin02[aa,mm]=dFtot02/\
			                                (qinf_tot*Slin02.S0.chord*alpha_eff)


			### Reference nonlinear solution
			Sref=uvlm.solver(M,Mw,b,Uinf,alpha_eff,rho)
			Sref.build_camber_plate(Mcamb=10,Pcamb=4)
			# solve
			Sref.solve_static_Gamma2d()
			# store
			Dnnl[aa,mm],Lnnl[aa,mm]=np.sum(Sref.FmatSta,0)
			CDnnl[aa,mm],CLnnl[aa,mm]=np.sum(Sref.FmatSta,0)/\
			                                     (qinf_tot*Sref.chord*alpha_eff)


		clist=['k','r','b','0.6',]
		### Aerodynamic forces
		fig1 = plt.figure('Drag',(12,4))
		fig2 = plt.figure('Lift',(12,4))
		ax1=fig1.add_subplot(111)
		ax2=fig2.add_subplot(111)

		ax1.plot(DAlphaList,Dlin01[:,mm],'k',lw=2,ls='--',
		                                label=r'M=%.2d lin 0 deg'%MList[mm])
		ax1.plot(DAlphaList,Dlin02[:,mm],'b',lw=2,ls=':',
		                               label=r'M=%.2d lin 10 deg'%MList[mm])
		ax1.plot(DAlphaList,Dnnl[:,mm],'r',lw=Nm-mm,
		                                      label=r'M=%.2d nnl'%MList[mm])
		#
		ax2.plot(DAlphaList,Llin01[:,mm],'k',lw=2,ls='--',
		                                label=r'M=%.2d lin 0 deg'%MList[mm])
		ax2.plot(DAlphaList,Llin02[:,mm],'b',lw=Nm-mm,ls=':',
		                               label=r'M=%.2d lin 10 deg'%MList[mm])
		ax2.plot(DAlphaList,Lnnl[:,mm],'r',lw=2,
		                                      label=r'M=%.2d nnl'%MList[mm])

		ax1.set_xlabel(r'\alpha [deg]')
		ax1.set_title(r'Drag')
		ax1.legend()
		ax2.set_xlabel(r'\alpha [deg]')
		ax2.set_title(r'Lift')
		ax2.legend()

		if self.SHOW_PLOT: plt.show()
		else: plt.close()




	def test_camber_aerofoil_speed(self):
		'''
		Calculate steady case / very short wake can be used. A cambered aerofoil,
		rotated of 0 and 10 deg w.r.t. the horizontal line, provides reference
		conditions for linearisation. Nonzero nodal velocities are enforced at 
		the aerofoil grid so as to obtain effective angles of attack between 
		0,20 deg.
		'''

		DAlphaList=[0.0,0.1,0.2,0.3,0.5,1.,2.,4.,6.,8.,
		            9.5,9.7,9.8,9.9,10.,10.1,10.2,10.3,10.5,12.,14.,16.,18.] # degs
		#DAlphaList=[5.0]

		MList=[20]
		Nm,mm=1,0
		Na=len(DAlphaList)

		CDlin01,CLlin01=np.zeros((Na,Nm)),np.zeros((Na,Nm))
		CDlin02,CLlin02=np.zeros((Na,Nm)),np.zeros((Na,Nm))
		CDnnl,CLnnl=np.zeros((Na,Nm)),np.zeros((Na,Nm))

		Llin01,Dlin01=np.zeros((Na,Nm)),np.zeros((Na,Nm))
		Llin02,Dlin02=np.zeros((Na,Nm)),np.zeros((Na,Nm))
		Lnnl,Dnnl=np.zeros((Na,Nm)),np.zeros((Na,Nm))

		# input:
		Mw=2
		M=MList[mm]
		chord=3.
		b=0.5*chord
		Uinf=np.array([20.,0.])
		rho=1.225

		# reference condition 1: aerofoil at 0 deg angle
		alpha01=0.*np.pi/180.
		S01=uvlm.solver(M,Mw,b,Uinf,alpha01,rho=1.225)
		S01.build_camber_plate(Mcamb=10,Pcamb=4)
		S01.solve_static_Gamma2d()
		Ftot0=np.sum(S01.FmatSta,0)

		# reference condition 2: aerofoil at 10 deg angle
		alpha02=10.*np.pi/180.
		S02=uvlm.solver(M,Mw,b,Uinf,alpha02,rho=1.225)
		S02.build_camber_plate(Mcamb=10,Pcamb=4)
		S02.solve_static_Gamma2d()


		print('Testing steady aerofoil for M=%d...' %M)

		def get_wvec(aeff,a0,ux,uz):

			# get free stream angle
			a=aeff-a0
			tga=np.tan(a)

			### build residual
			def Fres(x):
				'''@warning: uses function variables!'''
				wx,wz=x
				F=np.zeros((2,))
				F[0]=tga*(ux+wx)-(uz+wz)
				F[1]=wx**2+2.*wx*ux+wz**2+2.*uz*wz
				return F

			# solve nnl equations
			x0=np.zeros((2,))
			wvec=scopt.newton_krylov(Fres,x0)

			asol=np.tan((uz+wvec[1])/(ux+wvec[0]))
			assert (asol-(aeff-a0))**2<1e-3, 'Solution not found!'

			return wvec


		for aa in range(len(DAlphaList)):

			# Perturbation
			alpha_eff=np.pi/180.*DAlphaList[aa]
			qinf_tot=0.5*rho*(S01.Uabs)**2
			print('\tAlpha effective=%.2f deg' %DAlphaList[aa])

			### Linearised solution 01:
			# Perturb reference state
			Slin01=linuvlm.solver(S01)
			wvec=get_wvec(alpha_eff,alpha01,Uinf[0],Uinf[1])
			Slin01.dZetadt[:,:]=-wvec
			# solve
			Slin01.solve_static_Gamma2d()
			# store data
			Dlin01[aa,mm],Llin01[aa,mm]=np.sum(Slin01.FmatSta,0)
			dFtot01=np.sum(Slin01.FmatSta,0)-Ftot0
			CDlin01[aa,mm],CLlin01[aa,mm]=dFtot01/\
			                                (qinf_tot*Slin01.S0.chord*alpha_eff)

			### Linearised solution 02:
			# Perturb reference state
			Slin02=linuvlm.solver(S02)
			wvec=get_wvec(alpha_eff,alpha02,Uinf[0],Uinf[1])
			Slin02.dZetadt[:,:]=-wvec
			# solve
			Slin02.solve_static_Gamma2d()
			# store data
			Dlin02[aa,mm],Llin02[aa,mm]=np.sum(Slin02.FmatSta,0)
			dFtot02=np.sum(Slin02.FmatSta,0)-Ftot0
			CDlin02[aa,mm],CLlin02[aa,mm]=dFtot02/\
			                                (qinf_tot*Slin02.S0.chord*alpha_eff)


			### Reference nonlinear solution
			Sref=uvlm.solver(M,Mw,b,Uinf,alpha_eff,rho)
			Sref.build_camber_plate(Mcamb=10,Pcamb=4)
			# solve
			Sref.solve_static_Gamma2d()
			# store
			Dnnl[aa,mm],Lnnl[aa,mm]=np.sum(Sref.FmatSta,0)
			CDnnl[aa,mm],CLnnl[aa,mm]=np.sum(Sref.FmatSta,0)/\
			                                     (qinf_tot*Sref.chord*alpha_eff)


		clist=['k','r','b','0.6',]
		### Aerodynamic forces
		fig1 = plt.figure('Drag',(12,4))
		fig2 = plt.figure('Lift',(12,4))
		ax1=fig1.add_subplot(111)
		ax2=fig2.add_subplot(111)

		ax1.plot(DAlphaList,Dlin01[:,mm],'k',lw=2,ls='--',
		                                label=r'M=%.2d lin 0 deg'%MList[mm])
		ax1.plot(DAlphaList,Dlin02[:,mm],'b',lw=2,ls=':',
		                               label=r'M=%.2d lin 10 deg'%MList[mm])
		ax1.plot(DAlphaList,Dnnl[:,mm],'r',lw=Nm-mm,
		                                      label=r'M=%.2d nnl'%MList[mm])
		#
		ax2.plot(DAlphaList,Llin01[:,mm],'k',lw=2,ls='--',
		                                label=r'M=%.2d lin 0 deg'%MList[mm])
		ax2.plot(DAlphaList,Llin02[:,mm],'b',lw=Nm-mm,ls=':',
		                               label=r'M=%.2d lin 10 deg'%MList[mm])
		ax2.plot(DAlphaList,Lnnl[:,mm],'r',lw=2,
		                                      label=r'M=%.2d nnl'%MList[mm])

		ax1.set_xlabel(r'\alpha [deg]')
		ax1.set_title(r'Drag')
		ax1.legend()
		ax2.set_xlabel(r'\alpha [deg]')
		ax2.set_title(r'Lift')
		ax2.legend()
		plt.show()

		if self.SHOW_PLOT: plt.show()
		else: plt.close()





if __name__=='__main__':


	T=Test_dyn()
	T.setUp()
	#T.SHOW_PLOT=True

	## Flat around zero
	#T.test_flat_rotation_about_zero()

	## Flat flat with vertical displacements only
	#T.test_flat_vert_disp_about_zero()

	## Cambered aerofoil
	#T.test_camber_rotation()
	
	## Cambered aerofoil - external gust speed
	#T.test_camber_external_speed()

	## Cambered aerofoil - aerofoil speed
	T.test_camber_aerofoil_speed()

	### run all
	#unittest.main()

