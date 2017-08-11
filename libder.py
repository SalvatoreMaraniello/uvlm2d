'''

Library of formula for computing derivatives.

author: S. Maraniello
date: 11 Jul 2017
note: formula derived in sympy. see src/delop/linsym*.py modules

'''

import numpy as np
from IPython import embed


def der_WncV_dZeta(zeta01_x,zeta02_x,zeta01_y,zeta02_y,V0_x,V0_y):
	'''
	Derivative of velocity WncV perpendicular to panel w.r.t. panel vertices
	coordinates.

	Variables:
		V0_*: velocity at collocation point
		zeta01_*: coordinates of grid point zeta01 (ahead)
		zeta02_*: coordinates of second grid point (behind)

	The output is a 4 element array with derivatives w.r.t. (in order):
		zeta01_x,zeta02_x,zeta01_y,zeta02_y

	Details: src/develop/linsym_Wnc.py
	'''

	dz_x=zeta02_x-zeta01_x
	dz_y=zeta02_y-zeta01_y

	derWncV=np.array(
		[(-V0_y*(dz_x**2 + dz_y**2) - dz_x*(V0_x*dz_y - V0_y*dz_x))/(dz_x**2 + dz_y**2)**(3/2),
 		(V0_y*(dz_x**2 + dz_y**2) - dz_x*(-V0_x*dz_y + V0_y*dz_x))/(dz_x**2 + dz_y**2)**(3/2), 
 		(V0_x*(dz_x**2 + dz_y**2) - dz_y*(V0_x*dz_y - V0_y*dz_x))/(dz_x**2 + dz_y**2)**(3/2), 
 		(-V0_x*(dz_x**2 + dz_y**2) - dz_y*(-V0_x*dz_y + V0_y*dz_x))/(dz_x**2 + dz_y**2)**(3/2)]
		)

	return derWncV



def der_Wnc0AGamma_dZeta(zeta01,zeta02,zetaA,zetaB,zetaC,nvec,CF,gamma,
	                                                              allring=True):
	'''
	Derivatives of induced velocity at the collocation point zetaC. Only the 
	constribution associated to vortex gamma is included.

	Details: src/develop/linsym_aic.py
	'''

	zeta01_x,zeta01_y=zeta01
	zeta02_x,zeta02_y=zeta02

	zetaA_x,zetaA_y=zetaA	
	zetaB_x,zetaB_y=zetaB

	zetaC_x,zetaC_y=zetaC

	R01=zetaC-zeta01
	R02=zetaC-zeta02
	R01_x,R01_y=R01
	R02_x,R02_y=R02

	R01sq=np.sum(R01**2)
	R02sq=np.sum(R02**2)


	n0_x,n0_y=nvec

	if allring:
		# contribution of both segment 1 and 2
		Der=np.array([ 
			-CF*gamma*(R01sq*n0_y - (R01_x*n0_y - R01_y*n0_x)*2*R01_x)/R01sq**2 ,
			 CF*gamma*(R02sq*n0_y - (R02_x*n0_y - R02_y*n0_x)*2*R02_x)/R02sq**2 ,
			 CF*gamma*(R01sq*n0_x + (R01_x*n0_y - R01_y*n0_x)*2*R01_y)/R01sq**2 ,
			-CF*gamma*(R02sq*n0_x + (R02_x*n0_y - R02_y*n0_x)*2*R02_y)/R02sq**2 ,
			-CF*gamma*(R01sq**2*R02sq*n0_y - R01sq**2*(R02_x*n0_y - R02_y*n0_x)*\
				2*R02_x - R01sq*R02sq**2*n0_y + R02sq**2*(R01_x*n0_y - R01_y*n0_x)*2*R01_x)/(2*R01sq**2*R02sq**2) ,
			-CF*gamma*(R01sq**2*R02sq*n0_y - R01sq**2*(R02_x*n0_y - R02_y*n0_x)*\
				2*R02_x - R01sq*R02sq**2*n0_y + R02sq**2*(R01_x*n0_y - R01_y*n0_x)*2*R01_x)/(2*R01sq**2*R02sq**2) ,
			CF*gamma*(R01sq**2*R02sq*n0_x + R01sq**2*(R02_x*n0_y - R02_y*n0_x)*\
				2*R02_y - R01sq*R02sq**2*n0_x - R02sq**2*(R01_x*n0_y - R01_y*n0_x)*2*R01_y)/(2*R01sq**2*R02sq**2) ,
			CF*gamma*(R01sq**2*R02sq*n0_x + R01sq**2*(R02_x*n0_y - R02_y*n0_x)*\
				2*R02_y - R01sq*R02sq**2*n0_x - R02sq**2*(R01_x*n0_y - R01_y*n0_x)*2*R01_y)/(2*R01sq**2*R02sq**2) ,
			])
	else:
		# contribution of segment 1 only
		Der=np.array([
			CF*gamma*(-n0_y*R01sq + (R01_x*n0_y - R01_y*n0_x)*2*R01_x)/R01sq**2 ,
			0 ,
			CF*gamma*(n0_x*R01sq + (R01_x*n0_y - R01_y*n0_x)*2*R01_y)/R01sq**2 ,
			0 ,
			CF*gamma*(n0_y*R01sq - (R01_x*n0_y - R01_y*n0_x)*2*R01_x)/(2*R01sq**2) ,
			CF*gamma*(n0_y*R01sq - (R01_x*n0_y - R01_y*n0_x)*2*R01_x)/(2*R01sq**2) ,
			-CF*gamma*(n0_x*R01sq + (R01_x*n0_y - R01_y*n0_x)*2*R01_y)/(2*R01sq**2) ,
			-CF*gamma*(n0_x*R01sq + (R01_x*n0_y - R01_y*n0_x)*2*R01_y)/(2*R01sq**2) ,
			])

	return Der



def der_NormalArea_dZeta(zeta01,zeta02,dGamma):
	'''
	This is one of the 2 terms composing the added mass. As it does not depend on 
	the elements coordinates, so the input variables zeta01,02 are ficticious. 
	dGamma is the time derivative of circulation change at the ring.
	'''

	Der=-dGamma*np.array([[ 0., 0., 1.,-1.], 
		                  [-1., 1., 0., 0.]])

	return Der


def der_Fjouk_ind_zeta(zeta01,zeta02,zetaCA,zetaCB,CF,Gamma,gamma_s,allring=True):
	'''
	Derivatives of force due to the induced velocity at a segment of vertex 
	zetaCA=zetaCB. Only the constribution associated to vortex Gamma is included. 
	gamma_s is the net vorticity at the segment, which is assumed to be constant.

	If zeta01=zetaCA, the corresponding derivatives have infinite value.

	Details: src/develop/linsym_fjouk_ind.py
	'''

	R01=zetaCA-zeta01
	R02=zetaCB-zeta02

	### avoid warning messages:
	# when R01 or R02 are zero, the derivatives are discarded - as they have
	# infinite value. To avoid issuing a warning, zero is replaced by a very 
	# small number. 
	if np.linalg.norm(R01)<1e-16: R01=(1.+1e-15)*zetaCA-zeta01
	if np.linalg.norm(R02)<1e-16: R02=(1.+1e-15)*zetaCB-zeta02

	R01_x,R01_y=R01
	R02_x,R02_y=R02

	R01sq=np.sum(R01**2)
	R02sq=np.sum(R02**2)


	if allring:
		Der=np.array([[
			CF*Gamma*gamma_s*(2*R01_x**2 - R01sq)/R01sq**2, 
		   -CF*Gamma*gamma_s*(2*R02_x**2 - R02sq)/R02sq**2, 
		  2*CF*Gamma*R01_x*R01_y*gamma_s/R01sq**2, 
		 -2*CF*Gamma*R02_x*R02_y*gamma_s/R02sq**2, 
		   -CF*Gamma*gamma_s*(2*R01_x**2 - R01sq)/R01sq**2, 
		    CF*Gamma*gamma_s*(2*R02_x**2 - R02sq)/R02sq**2, 
		 -2*CF*Gamma*R01_x*R01_y*gamma_s/R01sq**2, 
		  2*CF*Gamma*R02_x*R02_y*gamma_s/R02sq**2
		  ],[
		  2*CF*Gamma*R01_x*R01_y*gamma_s/R01sq**2, 
		 -2*CF*Gamma*R02_x*R02_y*gamma_s/R02sq**2, 
		    CF*Gamma*gamma_s*(2*R01_y**2 - R01sq)/R01sq**2, 
		   -CF*Gamma*gamma_s*(2*R02_y**2 - R02sq)/R02sq**2, 
		 -2*CF*Gamma*R01_x*R01_y*gamma_s/R01sq**2, 
		  2*CF*Gamma*R02_x*R02_y*gamma_s/R02sq**2, 
		   -CF*Gamma*gamma_s*(2*R01_y**2 - R01sq)/R01sq**2, 
		    CF*Gamma*gamma_s*(2*R02_y**2 - R02sq)/R02sq**2
		    ]])
	else: # contribution of segment 1 only
		Der=np.array([[
			CF*Gamma*gamma_s*(R01_x**2 - R01_y**2)/R01sq**2, 
		    0, 
		  2*CF*Gamma*R01_x*R01_y*gamma_s/R01sq**2, 
		    0, 
		   -CF*Gamma*gamma_s*(R01_x**2 - R01_y**2)/R01sq**2, 
		    0, 
		 -2*CF*Gamma*R01_x*R01_y*gamma_s/R01sq**2, 
		    0
		  ],[
		  2*CF*Gamma*R01_x*R01_y*gamma_s/R01sq**2, 
		    0, 
		   -CF*Gamma*gamma_s*(R01_x**2 - R01_y**2)/R01sq**2, 
		    0, 
		 -2*CF*Gamma*R01_x*R01_y*gamma_s/R01sq**2, 
		    0, 
		    CF*Gamma*gamma_s*(R01_x**2 - R01_y**2)/R01sq**2, 
		    0]])

	return Der



def der_Fjouk_ind_zeta_by_gamma(zeta01,zetaC,CF,gamma01,gammaC):
	'''
	Derivatives of force at vertex zetaC due to the induced velocity produced by
	segment zeta01. gamma01 and gammaC are the net vorticity at the segments.

	Details: src/develop/linsym_fjouk_ind_bygamma.py
	'''

	R01=zetaC-zeta01

	### avoid warning messages:
	# when R01 or R02 are zero, the derivatives are discarded - as they have
	# infinite value. To avoid issuing a warning, zero is replaced by a very 
	# small number. 
	if np.linalg.norm(R01)<1e-16: R01=(1.+1e-15)*zetaC-zeta01

	R01_x,R01_y=R01
	R01sq=np.sum(R01**2)

	Der=-np.array([[
			CF*gamma01*gammaC*(2*R01_x**2 - R01sq)/R01sq**2, 
		  2*CF*R01_x*R01_y*gamma01*gammaC/R01sq**2, 
		   -CF*gamma01*gammaC*(2*R01_x**2 - R01sq)/R01sq**2, 
		 -2*CF*R01_x*R01_y*gamma01*gammaC/R01sq**2
		 ],[
		  2*CF*R01_x*R01_y*gamma01*gammaC/R01sq**2, 
		    CF*gamma01*gammaC*(2*R01_y**2 - R01sq)/R01sq**2, 
		 -2*CF*R01_x*R01_y*gamma01*gammaC/R01sq**2,
		   -CF*gamma01*gammaC*(2*R01_y**2 - R01sq)/R01sq**2
		]])

	return Der