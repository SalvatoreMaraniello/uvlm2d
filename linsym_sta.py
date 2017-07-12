'''
Analytical linearisation of static terms of 2D UVLM solver.


Sign convention:

Scalar quantities are all lower case, e.g. zeta
Arrays begin with upper case, e.g. Zeta_i
2 D Matrices are all upper case, e.g. AW, ZETA=[Zeta_i]
3 D arrays (tensors) will be labelled with a 3 in the name, e.g. A3

'''

import sympy as sm
#import sympy.tensor as smtens
import sympy.tensor.array as smarr



# coordinates of bound grid
Zeta_ix, Zeta_iy, Zeta_iz = sm.symbols('Zeta_ix Zeta_iy Zeta_iz', real=True)


# coordinates of wake grid
Zeta_wix, Zeta_wiy, Zeta_wiz = sm.symbols('Zeta_wix Zeta_wiy Zeta_wiz', real=True)

# collocaiton point
Zeta_cix, Zeta_ciy, Zeta_ciz = sm.symbols('Zeta_cix Zeta_ciy Zeta_ciz', real=True)






def biot_savart_2d(cv,zeta,gamma=1.0):
	drv=cv-zeta
	drabs=np.linalg.norm(drv)
	qv = 0.5*gamma/np.pi/drabs**2 * np.array([-drv[1],drv[0]])

	return qv


def cross_matrix(av):
	'''
	Builds the matrix A  from the vecotr av such that
		cv = av x bv = A bv
	'''
	ax,ay,az=av
	A=sm.Matrix( #smarr.Array(
		       [[  0,-az, ay],
		        [ az,  0,-ax],
		        [-ay, ax,  0], ])
	return A






if __name__=='__main__':


	# verify cross-product
	ax,ay,az=sm.symbols('ax ay az', real=True)
	av=smarr.Array([ax,ay,az])
	av=sm.Matrix([ax,ay,az])
	A=cross_matrix(av)

	bv=sm.MatMul(A,av)




