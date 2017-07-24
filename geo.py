'''
2D UVLM solver
author: S. Maraniello
date: 13 Jul 2017

Geometry module: collection of methods to set-up aerofoil geometry

'''


import numpy as np
from IPython import embed



def build_flat_plate(b,alpha,Uinf,K,Kw,perc_ring):
	''' 
	Build geometry/flow field of a flat plate at an angle alpha w.r.t the global 
	frame  Oxy.  

	@warning: alpha is not necessarely the angle between plate and velocity;
	@warning: if the aerofoil has a velocity, the wake should be displaced. 
	This effect is not accounted here as in the static solution the wake has 
	no impact.
	'''

	# params
	Ndim=2
	M=K-1
	Mw=Kw-1

	Rmat=np.zeros((K,Ndim))
	# grid coordinates
	Zeta=np.zeros((K,Ndim))
	ZetaW=np.zeros((Kw,Ndim))
	Uzeta = np.zeros((K,Ndim))

	# def wing panels coordinates
	rLE=2.*b*np.array([-np.cos(alpha),np.sin(alpha)])
	rTE=np.zeros((Ndim,))
	for ii in range(Ndim):
		Rmat[:,ii]=np.linspace(rLE[ii],rTE[ii],K)

	# def bound vortices "rings" coordinates
	dRmat=np.diff(Rmat.T).T
	Zeta[:K-1,:]=Rmat[:K-1,:]+perc_ring*dRmat
	Zeta[-1,:]=Zeta[-2,:]+dRmat[-1,:]

	# def wake vortices coordinates
	dl=2.*b/M
	twake=Uinf/np.linalg.norm(Uinf)
	EndWake=Zeta[-1,:]+Mw*dl*twake
	for ii in range(Ndim):
		ZetaW[:,ii]=np.linspace(Zeta[-1,ii],EndWake[ii],Kw)

	# set background velocity at vortex grid
	for nn in range(K): 
		Uzeta[nn,:]=Uinf

	return Rmat,Zeta,ZetaW,Uzeta



def build_camber_plate(b,Mcamb,Pcamb,alpha,Uinf,K,Kw,perc_ring):
	''' 
	Build geometry/flow field of a cambered plate at an angle alpha w.r.t the 
	global frame  Oxy.

	Mcamb and Pcamb are the maximum camber (in percentage of chord) and the
	position of max. camber (in 10s of chord). These geometry is built following
	the convention for NACA 4 digit aerofoils.

	@warning: alpha is not necessarely the angle between plate and velocity;
	@warning: if the aerofoil has a velocity, the wake should be displaced. 
	This effect is not accounted here as in the static solution the wake has 
	no impact.
	'''

	# params
	Ndim=2
	M,Mw=K-1,Kw-1

	Rmat=np.zeros((K,Ndim))
	# grid coordinates
	Zeta=np.zeros((K,Ndim))
	ZetaW=np.zeros((Kw,Ndim))
	Uzeta = np.zeros((K,Ndim))

	# NACA camber gemetry
	xv=np.linspace(0.,1.,K)
	yv=np.zeros((K,))
	iifore=xv<=0.1*Pcamb
	iiaft=iifore-True
	m,p=1e-2*Mcamb,1e-1*Pcamb
	yv[iifore]=m/p**2*xv[iifore]*(2.*p-xv[iifore])
	yv[iiaft]=m/(1.-p)**2*(1.-2.*p+2.*p*xv[iiaft]-xv[iiaft]**2)
	xv=2.*b*(xv-1.)
	yv=2.*b*yv
	# import matplotlib.pyplot as plt
	# plt.plot(xv,yv)
	# plt.show()
	# embed()

	# Rotate
	sn,cs=np.sin(-alpha),np.cos(-alpha)
	Rot=np.array([[cs,-sn], [sn, cs]])
	for kk in range(K):
		Rmat[kk,:]=np.dot(Rot,[xv[kk],yv[kk]])

	# def bound vortices "rings" coordinates
	dRmat=np.diff(Rmat.T).T
	Zeta[:K-1,:]=Rmat[:K-1,:]+perc_ring*dRmat
	Zeta[-1,:]=Zeta[-2,:]+dRmat[-1,:]

	# def wake vortices coordinates
	dl=2.*b/M
	twake=Uinf/np.linalg.norm(Uinf)
	EndWake=Zeta[-1,:]+Mw*dl*twake
	for ii in range(Ndim):
		ZetaW[:,ii]=np.linspace(Zeta[-1,ii],EndWake[ii],Kw)

	# set background velocity at vortex grid
	for nn in range(K): 
		Uzeta[nn,:]=Uinf

	return Rmat,Zeta,ZetaW,Uzeta



def rotate_aerofoil(Zeta0,dalpha):
	'''
	given an aerofoil of coordinates Zeta, the function rotates it of an angle 
	dalpha (positive when E moves upward). The rotation is about the trailing
	edge.
	'''

	K=Zeta0.shape[0]
	Zeta=0.0*Zeta0

	# Rotate
	sn,cs=np.sin(-dalpha),np.cos(-dalpha)
	Rot=np.array([[cs,-sn], [sn, cs]])
	for kk in range(K):
		Zeta[kk,:]=np.dot(Rot,Zeta0[kk,:])

	return Zeta



if __name__=='__main__':
	
	import matplotlib.pyplot as plt

	# build cambered aerofoil
	Rmat0,Zeta0,ZetaW0,Uzeta0=build_camber_plate(
					b=2.,Mcamb=20,Pcamb=.5,alpha=30.*np.pi/180.,
											  Uinf=10.,K=101,Kw=11,perc_ring=.25)

	# rotate it
	Zeta=rotate_aerofoil(Zeta0,dalpha=30.*np.pi/180.)

	plt.plot(Zeta0[:,0],Zeta0[:,1],'r',label='reference')
	plt.plot(Zeta[:,0],Zeta[:,1],'b',label='rotated')
	plt.legend()
	plt.show()

