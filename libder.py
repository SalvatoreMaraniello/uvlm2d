'''

Library of formula for computing derivatives.

author: S. Maraniello
date: 11 Jul 2017
note: formula derived in sympy. see src/delop/linsym*.py modules

'''

import numpy as np


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



