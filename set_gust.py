'''
@author: salvatore maraniello
@date: 26 May 2017
@brief: collection of methods to define gust input for dynamic analysis. All 
methods take in  input a class uvlm2d_dyn.
@note: all methods assume the geometry of the aerofoil at t=0 s is already built

References:
[1] Aeroelasticity, MSc notes (part 2)
'''

import numpy as np
import matplotlib.pyplot as plt
from IPython import embed


def sin(S,w0,L,ImpStart=False):
	'''
	Defines a sinusoidal gust with only vertical component moving horizontally
	at velocity S.Uinf[0]/S.S0.Uinf[0].
	The method accept an instance of both the linear and geometrically-exact
	solvers, lin_uvlm2d_dym.solver and uvlm2d_dym.solver.

	@warning: the gust shape only depends on the horizontal coordinates of the
	aerofoil. The x coordinates of the aerofoil and its wake are assumed not to 
	vary in time.

	'''	

	### set gust parameters
	C=2.*np.pi/L
	if hasattr(S,'Uinf'): Ux=S.Uinf[0]
	else: Ux=S.S0.Uinf[0]

	### rescale grid x coordinates at t=0
	if hasattr(S,'ZetaW'): # exact sol.
		xcoord=np.concatenate( (S.Zeta[:,0],S.ZetaW[:,0]) ) - S.Zeta[0,0]
	else: # linearised model
		xcoord=S.Zeta[:,0]-S.Zeta[0,0]

	### set profile
	for tt in range(S.NT):
		wgust = w0*np.sin( C*(Ux*S.time[tt] - xcoord) )

		if hasattr(S,'THWzetaW'):
			S.THWzeta[tt,:,1]=wgust[:K]
			S.THWzetaW[tt,:,1]=wgust[K:]
		else:
			S.THWzeta[tt,:,1]=wgust

	if ImpStart==False:
		S.Wzeta[:,1]=S.THWzeta[0,:,1]

	return S



def visualise_gust(S,N):
	'''
	Visualise the gust at N equally spaced snapshots
	'''

	fig = plt.figure('Gust profile',(10,6))
	ax = fig.add_subplot(111)

	for tt in range(0,S.NT,int(S.NT/N)):
		xv=np.concatenate( (S.Zeta[:,0],S.ZetaW[1:,0]) )
		wv=np.concatenate( (S.THWzeta[tt,:,1],S.THWzetaW[tt,1:,1]) )
		ax.plot(xv,wv,label='t=%.3f'%S.time[tt])

	ax.legend()




if __name__=='__main__':


	import uvlm2d_dyn as uvlm
	import pp_uvlm2d as pp
	import analytical as an


	# random geometry
	c=3.
	b=0.5*c
	uinf=2.0
	T=2.0

	M=400
	WakeFact=3
	TimeList=[]
	THCFList=[]

	# build solver class
	S=uvlm.solver(T=T,M=M,Mw=M*WakeFact,b=b,Uinf=np.array([uinf,0.]),
                  alpha=0.*np.pi/180.,rho=1.225)
	S.build_flat_plate()
	S=sin(S,w0=.4,L=2.0*S.chord)

	### visualise gust
	# time to cover 1 chord
	tc=S.chord/S.Uinf[0]
	# gust moves 1 chord between intervals
	visualise_gust(S,N=int(T/tc))
	plt.show()

