'''
author: S. Maraniello
date: 12 Jul 2017

Functions for symbolic manipulation

'''


import sympy as sm
#import sympy.tensor as smtens
import sympy.tensor.array as smarr
from IPython import embed




def scalar_product(Av,Bv):
	'''
	Scalar product for sympy.tensor.array
	'''

	N=len(Av)
	assert N==len(Bv), 'Array dimension not matching'

	P=Av[0]*0
	for ii in range(len(Av)):
		P+=Av[ii]*Bv[ii]

	return P


def matrix_product(Asm,Bsm):
	'''
	Matrix product between 2D sympy.tensor.array
	'''

	assert len(Asm.shape)<3 or len(Bsm.shape)<3,\
	                  'Attempting matrix product between 3D (or higher) arrays!'
	assert Asm.shape[1]==Bsm.shape[0], 'Matrix dimensions not compatible!'

	P=smarr.tensorproduct(Asm,Bsm)
	Csm=smarr.tensorcontraction(P,(1,2))

	return Csm


def simplify(Av):
	'''
	Simplify each element of matrix/array
	'''

	Av_simple=[]

	if len(Av.shape)==1:
		for ii in range(len(Av)):
			Av_simple.append(Av[ii].simplify())

	elif len(Av.shape)==2:
		for ii in range(Av.shape[0]):
			row=[]
			for jj in range(Av.shape[1]):
				row.append(Av[ii,jj].simplify())
			Av_simple.append(row)

	else:
		raise NameError('Method not developed for 3D arrays!')

	return smarr.MutableDenseNDimArray(Av_simple)


def subs(Av,expr_old,expr_new):
	'''
	Iteratively apply the subs method to each element of tensor.
	'''

	Av_sub=[]

	if len(Av.shape)==1:
		for ii in range(len(Av)):
			Av_sub.append(Av[ii].subs(expr_old,expr_new))

	elif len(Av.shape)==2:
		for ii in range(Av.shape[0]):
			row=[]
			for jj in range(Av.shape[1]):
				row.append(Av[ii,jj].subs(expr_old,expr_new))
			Av_sub.append(row)

	else:
		raise NameError('Method not developed for 3D arrays!')

	return smarr.MutableDenseNDimArray(Av_sub)


def scalar_deriv(a,xList):
	'''Compute derivatives of a scalar w.r.t. a list of valirables'''

	Nx=len(xList)
	Der=[]
	for ii in range(Nx):
		Der.append(a.diff(xList[ii]))

	return smarr.MutableDenseNDimArray(Der)




# def skew_matrix(Av):
# 	'''
# 	Builds the matrix A  from the vecotr av such that
# 		cv = av x bv = A bv
# 	'''
# 	ax,ay,az=Av
# 	A=sm.Matrix( #smarr.Array(
# 		       [[  0,-az, ay],
# 		        [ az,  0,-ax],
# 		        [-ay, ax,  0], ])
# 	return A


if __name__=='__main__':

	import numpy as np

	### check matrix product
	Alist=[[1,4],[2,5],[7,4],[1,9]]
	Blist=[[5,8,4],[1,9,7]]
	C=np.dot( np.array(Alist),np.array(Blist) )

	Asm=smarr.Array(Alist)
	Bsm=smarr.Array(Blist)
	Csm=matrix_product(Asm,Bsm)


	### check simplify
	Csm_simple=simplify(Csm)


	# # verify cross-product
	# ax,ay,az=sm.symbols('ax ay az', real=True)
	# av=smarr.Array([ax,ay,az])
	# av=sm.Matrix([ax,ay,az])
	# A=cross_matrix(av)
	# bv=sm.MatMul(A,av)





	embed()
	