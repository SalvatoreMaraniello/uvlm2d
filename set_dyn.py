'''
@author: salvatore maraniello
@date: 22 May 2017
@brief: collection of methods to define dynamic cases. All methods take in 
input a class uvlm2d_dyn.
@note: all methods assume the geometry of the aerofoil at t=0 s is already built.

References:
[1] Simpson, R.J.S., Palacios, R. & Murua, J., 2013. Induced-Drag 
	Calculations in the Unsteady Vortex Lattice Method. AIAA Journal, 51(7), 
	pp.1775–1779.
'''

import numpy as np
import matplotlib.pyplot as plt





def plunge(S,f0,H):
	'''
	Set aerofoil plunge motion of frequency f0 (Hz) and amplitude H such that:
		Zeta[ii,1](t) = Zeta[ii,1](t=0) + H*[1-cos(2*pi*f0*t)]
	which is as per Ref.1, except for the constant term.
	@note: the method will update the vertical coordinate of the aerofoil t each
	time-step

	@warning: the initial shape of the wake is flat. To improve convergence
	an initial sinusoidal wake should be assumed
	'''

	# def. position
	S.THZeta[0,:,:]=S.Zeta
	for tt in range(1,S.NT):
		S.THZeta[tt,:,0]=S.Zeta[:,0]
		S.THZeta[tt,:,1]=S.Zeta[:,1]+H*(1.-np.cos(2.*np.pi*f0*S.time[tt]))
	# def. velocity at t=0
	S.dZetadt[:,1]=0.0

	### sinusoidal version
	### Zeta[ii,1](t) = Zeta[ii,1](t=0) + H*sin(2*pi*f0*t)
	# # def. position
	# S.THZeta[0,:,:]=S.Zeta
	# for tt in range(1,S.NT):
	# 	S.THZeta[tt,:,0]=S.Zeta[:,0]
	# 	S.THZeta[tt,:,1]=S.Zeta[:,1]+H*np.sin(2.*np.pi*f0*S.time[tt])
	# # def. velocity at t=0
	# S.dZetadt[:,1]=2.*np.pi*f0*H

	### @todo: def wake initial shape

	return S




def visualise_motion(S,Wake=True):
	'''
	Visualise time histories of leading edge, trailing edge and mid-point of the
	aerofoil.
	'''

	if S.K>2: midnode=int(S.K/2.)

	fig = plt.figure('Aerofoil motion',(10,6))

	ax = fig.add_subplot(121)
	ax.set_title(r'Horizontal')
	ax.plot(S.time,S.THZeta[:,0,0],'b',label=r'LE')
	if S.K>2: ax.plot(S.time,S.THZeta[:,midnode,0],'r',label=r'Node %d'%midnode)
	ax.plot(S.time,S.THZeta[:,S.K-1,0],'k',label=r'TE')
	ax.legend()

	ax = fig.add_subplot(122)
	ax.set_title(r'Vertical')
	ax.plot(S.time,S.THZeta[:,0,1],'b',label=r'LE')
	if S.K>2: ax.plot(S.time,S.THZeta[:,midnode,1],'r',label=r'Node %d'%midnode)
	ax.plot(S.time,S.THZeta[:,S.K-1,1],'k',label=r'TE')
	ax.legend()


	if Wake:
		if S.Kw>2: midnode=int(S.Kw/2.)

		figw = plt.figure('wake motion',(10,6))

		axw = figw.add_subplot(121)
		axw.set_title(r'Horizontal')
		axw.plot(S.time,S.THZetaW[:,0,0],'b',label=r'TE')
		if S.Kw>2: axw.plot(S.time,S.THZetaW[:,midnode,0],'r',label=r'Node %d'%midnode)
		axw.plot(S.time,S.THZetaW[:,S.K-1,0],'k',label=r'end')
		axw.legend()

		axw = figw.add_subplot(122)
		axw.set_title(r'Vertical')
		axw.plot(S.time,S.THZetaW[:,0,1],'b',label=r'TE')
		if S.Kw>2: axw.plot(S.time,S.THZetaW[:,midnode,1],'r',label=r'Node %d'%midnode)
		axw.plot(S.time,S.THZetaW[:,S.Kw-1,1],'k',label=r'end')
		axw.legend()		


	return None















