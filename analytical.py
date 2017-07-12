'''
@author: salvatore maraniello
@date: 23 May 2017
@brief: Analytical solutions for 2D aerofoil based on thin plates theory
@note: 

References:
[1] Simpson, R.J.S., Palacios, R. & Murua, J., 2013. Induced-Drag 
	Calculations in the Unsteady Vortex Lattice Method. AIAA Journal, 51(7), 
	pp.1775–1779.
[2] Gulcat, U., 2009. Propulsive Force of a Flexible Flapping Thin Airfoil. 
	Journal of Aircraft, 46(2), pp.465–473.
'''



import numpy as np
import scipy.special as scsp
from IPython import embed

# imaginary variable
j=1.0j


def theo_fun(k):
	'''Returns Theodorsen function at a reduced frequency k'''

	H1=scsp.hankel2(1,k)
	H0=scsp.hankel2(0,k)

	C=H1/(H1+j*H0)

	return C


def theo_lift(w,A,H,c,rhoinf,uinf,x12):
	'''
	Theodorsen's solution for lift of aerofoil undergoing sinusoidal motion:
		w: frequency (rad/sec) of oscillation
		A: amplitude of angle of attack change
		H: amplitude of plunge motion
		c: aerofoil chord
		rhoinf: flow density
		uinf: flow speed
		x12: distance of elastic axis from mid-point of aerofoil (positive if 
		    the elastic axis is ahead)

	Time histories are built assuming 
		a(t)=+/- A cos(w t) ??? not verified
		h(t)=-H cos(w t)
	'''

	# reduced frequency
	k=0.5*w*c/uinf

	# compute theodorsen's function
	Ctheo=theo_fun(k)

	# Lift: circulatory
	Lcirc=np.pi*rhoinf*uinf*c*Ctheo*( (uinf+w*j*(0.25*c+x12))*A +w*H*j)
	Lmass=0.25*np.pi*rhoinf*c**2*( (j*w*uinf-x12*w**2)*A -H*w**2 )
	Ltot=Lcirc+Lmass

	return Ltot, Lcirc, Lmass


def garrick_drag_plunge(w,H,c,rhoinf,uinf,time):
	'''
	Returns Garrick solution for drag coefficient at a specific time.
	Ref.[1], eq.(8) (see also eq.(1) and (2)) or Ref[2], eq.(2)
	The aerofoil vertical motion is assumed to be:
		h(t)=-H*cos(wt)
	The Cd is such that:
		Cd>0: drag
		Cd<0: suction
	'''

	b=0.5*c
	k=b*w/uinf
	Hast=H/b
	s=uinf*time/b

	# compute theodorsen's function
	Ctheo=theo_fun(k)

	Cd=-2.*np.pi*k**2 *Hast**2 *(
	                          Ctheo.imag*np.cos(k*s)+Ctheo.real*np.sin(k*s) )**2

	return Cd


def garrick_drag_pitch(w,A,c,rhoinf,uinf,x12,time):
	'''
	Returns Garrick solution for drag coefficient at a specific time.
	Ref.[1], eq.(9), (10) and (11)
	The aerofoil pitching motion is assumed to be:
		a(t)=A*sin(wt)=A*sin(ks)
	The Cd is such that:
		Cd>0: drag
		Cd<0: suction
	'''

	x12=x12/c
	b=0.5*c
	k=b*w/uinf
	s=uinf*time/b

	# compute theodorsen's function
	Ctheo=theo_fun(k)
	F,G=Ctheo.real,Ctheo.imag
	sks,cks=np.sin(k*s),np.cos(k*s)

	# angle of attack
	a=A*sks

	# lift term
	Cl=np.pi*A* ( k*cks 
		        + x12*k**2*sks 
		        + 2.*F*( sks+(0.5-x12)*k*cks ) 
		        + 2.*G*( cks-(0.5-x12)*k*sks ) )

	# suction force
	Y1=2.*(F-k*G*(0.5-x12))
	Y2=2.*(G-k*F*(0.5-x12))-k
	Cs=0.5*np.pi*A**2 * (Y1*sks+Y2*cks)**2

	Cd=a*Cl-Cs

	return Cd


def sears_lift_sin_gust(w0,L,Uinf,chord,tv):
	'''
	Returns the lift coefficient for a sinusoidal gust (see set_gust.sin).
	'''

	# reduced frequency
	kg=np.pi*chord/L
	# Theo's funciton
	Ctheo=theo_fun(kg)
	# Sear's function
	J0,J1=scsp.j0(kg),scsp.j1(kg)
	S= (J0-1.0j*J1)*Ctheo + 1.0j*J1


	phase=np.angle(S)
	CL=2.*np.pi*w0/Uinf * np.abs(S) * np.sin(2.*np.pi*Uinf/L*tv + phase)

	return CL


def wagner_imp_start(aeff,Uinf,chord,tv):
	'''
	Lift coefficient resulting from impulsive start solution. 
	'''

	sv=2.0*Uinf/chord*tv
	fiv=1.0-0.165*np.exp(-0.0455*sv)-0.335*np.exp(-0.3*sv)
	CLv=2.*np.pi*aeff*fiv

	return CLv




if __name__=='__main__':
	
	import matplotlib.pyplot as plt

	### geometry
	c=3.#m
	b=0.5*c

	### motion
	ktarget=1.
	H=0.02*b #m Ref.[1]
	A=1.*np.pi/180.#rad - Ref.[1]
	x12=-0.5*c

	f0=5.#Hz
	w0=2.*np.pi*f0 #rad/s

	uinf=b*w0/ktarget
	rhoinf=1.225 #kg/m3
	qinf=0.5*c*rhoinf*uinf**2

	#C=theo_fun(k=ktarget)
	#L=theo_lift(w0,A,H,c,rhoinf,uinf,x12)


	##### Plunge Induced drag
	Ncicles=5
	tv=np.linspace(0.,2.*np.pi*Ncicles/w0,200*Ncicles+1)
	Cdv=garrick_drag_plunge(w0,H,c,rhoinf,uinf,tv)
	hv=-H*np.cos(w0*tv)
	dhv=w0*H*np.sin(w0*tv)
	aeffv=np.arctan(-dhv/uinf)
	# fig = plt.figure('Induced drag - plunge motion',(10,6))
	# ax=fig.add_subplot(111)
	# ax.plot(tv,hv/c,'r',label=r'h/c')
	# ax.plot(tv,Cdv,'k',label=r'Induced Drag')
	# ax.legend()
	# plt.show()
	fig = plt.figure('Plunge motion - Phase vs kinematics',(10,6))
	ax=fig.add_subplot(111)
	#ax.plot(aeffv,hv/c,'r',label=r'h/c')
	ax.plot(180./np.pi*aeffv,Cdv,'k',label=r'Induced Drag')
	ax.set_xlabel('deg')
	ax.legend()
	plt.close()


	##### Pitching Induced drag
	Ncicles=5
	tv=np.linspace(0.,2.*np.pi*Ncicles/w0,200*Ncicles+1)
	Cdv=garrick_drag_pitch(w0,A,c,rhoinf,uinf,x12,tv)
	aeffv=A*np.sin(w0*tv)
	fig = plt.figure('Pitch motion - Phase vs kinematics',(10,6))
	ax=fig.add_subplot(111)
	#ax.plot(aeffv,hv/c,'r',label=r'h/c')
	ax.plot(180./np.pi*aeffv,Cdv,'k',label=r'Induced Drag')
	ax.set_xlabel('deg')
	ax.legend()



	##### Sear's solution test
	L=.5*c
	w0=0.3
	uinf=6.0

	# gust profile at LE
	tv=np.linspace(0.,2.,300)
	C=2.*np.pi/L
	wgustLE = w0*np.sin( C*uinf*tv )
	CLv = sears_lift_sin_gust(w0,L,uinf,c,tv)

	fig = plt.figure('Gust response',(10,6))
	ax=fig.add_subplot(111)
	ax.plot(tv,wgustLE,'k',label=r'vertical gust velocity at LE [m/s]')
	ax.plot(tv,CLv,'r',label=r'CL')
	ax.set_xlabel('time')
	ax.legend()
	#plt.show()
	plt.close('all')



	##### Wagner impulsive start
	uinf=20.0
	chord=3.0
	aeff=2.0*np.pi/180.
	tv=np.linspace(0.,10.,300)

	CLv=wagner_imp_start(aeff,uinf,chord,tv)
	CLv_inf=wagner_imp_start(aeff,uinf,chord,1e3*tv[-1])

	fig = plt.figure('Impulsive start',(10,6))
	ax=fig.add_subplot(111)
	ax.plot(tv,CLv/CLv_inf,'r',label=r'CL')
	ax.set_xlabel('time')
	ax.legend()
	plt.show()
	embed()









