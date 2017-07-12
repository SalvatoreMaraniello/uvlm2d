'''
Investigate effect of different methods for computing force
'''



import numpy  as np
import matplotlib.pyplot as plt

import uvlm2d_dyn as uvlm
import pp_uvlm2d as pp
import analytical as an
import set_dyn, set_gust
import read
from IPython import embed


##### Induced drag comparison
#
filename='./res_plunge/k10/M20wk30cyc16'
savefold='./figs_plunge/k10/'
#
#filename='./res_plunge/k05/M20wk30cyc12'
#savefold='./figs_plunge/k05/'
#
#filename='./res_plunge/k01/M16wk20cyc10'
#savefold='./figs_plunge/k01/'
#
#ktarget=0.1*np.float(filename[14:16])


##### Lift comparison
#filename='./res_plunge/k02/M16wk20cyc10'
#ktarget=0.25
filename='./res_plunge/k05/M40wk20cyc12' #### HF 
ktarget=0.5
#filename='./res_plunge/k07/M16wk20cyc10'  ####<----- HF launched
#ktarget=0.75


S=read.h5file(filename).solver


################################################################################
#
### Analytical solution
#
#####
w0=ktarget*S.Uabs/S.b
H=.5*np.max(S.THZeta[:,:,1])

# Garrik - induced drag
Cdv_an=an.garrick_drag_plunge(w0,H,S.chord,S.rho,S.Uabs,S.time)
hv_an=-H*np.cos(w0*S.time)
dhv=w0*H*np.sin(w0*S.time)
aeffv_an=np.arctan(-dhv/S.Uabs)

# Theodorsen - lift
Ltot_an,Lcirc_an,Lmass_an=an.theo_lift(w0,0,H,S.chord,S.rho,S.Uinf[0],0.0)
ph_tot,ph_circ,ph_mass=np.angle(Ltot_an),np.angle(Lcirc_an),np.angle(Lmass_an)
hc_an=hv_an/S.chord
CLtot_an=np.abs(Ltot_an)*np.cos(w0*S.time+ph_tot)/(S.chord*S.qinf)
CLcirc_an=np.abs(Lcirc_an)*np.cos(w0*S.time+ph_circ)/(S.chord*S.qinf)
CLmass_an=np.abs(Lmass_an)*np.cos(w0*S.time+ph_mass)/(S.chord*S.qinf)



################################################################################
#
### Recompute forces
#
### This is to make sure we do not mess up. 
#
### Remiders:
# - Joukovski force neglects TE vortex 
# - 

Fjouk=np.zeros((S.M,2))
Fmass=np.zeros((S.M,2))
Ftot_jouk=np.zeros((S.NT,2))
Ftot_mass=np.zeros((S.NT,2))
DZeta=np.diff(S.Zeta.T).T # constant
Nmat=S.Nmat


for tt in range(1,S.NT):
	# induced velocity at the grid
	Vtot_zeta=S.THVtot_zeta[tt,:,:]
	gamma=S.THgamma[tt,:]
	Gamma=S.THGamma[tt,:]
	### Force static - Joukovski - at segments=first M grid points
	for nn in range(S.M):
		Fjouk[nn,:]=-S.rho*gamma[nn]*\
		                            np.array([-Vtot_zeta[nn,1],Vtot_zeta[nn,0]])
	# Force dynamic - added mass - collocation point
	for nn in range(S.M):
		Fmass[nn,:]=-S.rho*np.linalg.norm(DZeta[nn,:])*\
		                   (S.THGamma[tt,nn]-S.THGamma[tt-1,nn])/S.dt*Nmat[nn,:]
	# Total forces
	for ii in range(2):
		Ftot_jouk[tt,ii]=np.sum(Fjouk[:,ii])
		Ftot_mass[tt,ii]=np.sum(Fmass[:,ii])
Ftot=Ftot_jouk+Ftot_mass




THCF=Ftot/S.qinf/S.chord
#THCF=S.THFaero/S.qinf/S.chord

















##### Post-process numerical solution
hc_num=(S.THZeta[:,0,1]-H)/S.chord
#aeffv_num=aeffv_an
aeffv_num=0.0*aeffv_an # derivative is the same as analytical
for tt in range(1,S.NT):
	#aeffv_num[tt]=-((S.THZeta[tt,0,1]-S.THZeta[tt-1,0,1])/S.dt/S.Uinf[0])
	aeffv_num[tt]=-np.arctan((S.THZeta[tt,0,1]-S.THZeta[tt-1,0,1])/S.dt/
		                                                              S.Uinf[0])

# Mass and circulatory contribution
THCFmass=np.zeros((S.NT,2))
for tt in range(S.NT):
	THCFmass[tt,:]=S.THFmassC[tt,:,:].sum(0)/S.qinf/S.chord
THCFcirc=THCF-THCFmass
# # Approximation of added mass - verified
# THCFmass_approx=np.zeros((S.NT))
# Gtot_old=0.0
# for tt in range(1,S.NT):
# 	Gtot_new=S.THGamma[tt,:].sum()
# 	dGtot=Gtot_new-Gtot_old
# 	Gtot_old=Gtot_new
# 	THCFmass_approx[tt]=S.rho*dGtot/S.dt/S.qinf/S.M


fig = plt.figure('Lift in plunge motion - Phase vs kinematics',(10,6))
ax=fig.add_subplot(111)
# numerical
ax.plot(hc_num,THCF[:,1],'k',label=r'Num Tot')
#ax.plot(hc_num,THCFmass[:,1],'b',label=r'Num Mass')
#ax.plot(hc_num,THCFcirc[:,1],'r',label=r'Num Jouk')
# analytical
ax.plot(hc_an,CLtot_an,'k--',lw=2,marker='o',markevery=(.3),label=r'An Tot')
#ax.plot(hc_an,CLmass_an,'b--',lw=2,marker='o',markevery=(.3),label=r'An Mass')
#ax.plot(hc_an,CLcirc_an,'r--',lw=2,marker='o',markevery=(.3),label=r'An Circ')
ax.set_xlabel('h/c')
ax.set_ylim(-.8,.8)
ax.legend()


fig = plt.figure('Induced drag in plunge motion - Phase vs kinematics',(10,6))
ax=fig.add_subplot(111)
ax.plot(180./np.pi*aeffv_an,Cdv_an,'k',label=r'Analytical')
ax.plot(180./np.pi*aeffv_num,THCF[:,0],'b',label=r'Numerical')
ax.set_xlabel('deg')
ax.legend()
##### The delay due to backward differences matters
# fig = plt.figure('Effectiver angles',(10,6))
# ax=fig.add_subplot(111)
# ax.plot(S.time,180./np.pi*aeffv_an,'k',label=r'Analytical')
# ax.plot(S.time,180./np.pi*aeffv_num,'b',label=r'Numerical')
# ax.set_xlabel('sec')
# ax.set_ylabel('deg')
# ax.legend()



fig = plt.figure('Lift coefficient',(10,6))
ax=fig.add_subplot(111)
# effective angle of attack
ax.plot(S.time,hc_num,'y',lw=1,label='h/c')
ax.plot(S.time,aeffv_num,'0.6',lw=2,label='Angle of attack [10 deg]')
# numerical
ax.plot(S.time, THCF[:,1],'k',lw=1,label='Num Tot')
#ax.plot(S.time, THCFmass[:,1],'b',lw=1,label='Num Mass')
#ax.plot(S.time, THCFcirc[:,1],'r',lw=1,label='Num Jouk')
## approximation
#ax.plot(S.time, THCFmass_approx, 'y^', label='Approx Mass')
# Theodorsen
ax.plot(S.time, CLtot_an,'k--',lw=2,label='An Tot')
#ax.plot(S.time, CLmass_an,'b--',lw=2,label='An Mass')
#ax.plot(S.time, CLcirc_an,'r--',lw=2,label='An Circ')
ax.set_xlabel('time')
ax.set_title('Lift coefficient')
ax.legend()


fig = plt.figure('Drag coefficient',(10,6))
ax=fig.add_subplot(111)
# numerical
ax.plot(S.time, THCF[:,0],'k',lw=1,label='Num Tot')
#ax.plot(S.time, THCFmass[:,0],'b',lw=1,label='Num Mass')
#ax.plot(S.time, THCFcirc[:,0],'r',lw=1,label='Num Jouk')
# Garrik
ax.plot(S.time, Cdv_an,'k--',lw=2,label='An Tot')
ax.set_xlabel('time')
ax.set_title('Drag coefficient')
ax.legend()



plt.show()
