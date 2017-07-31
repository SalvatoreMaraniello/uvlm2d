'''
Analytical linearisation of added mass terms

Sign convention:

Scalar quantities are all lower case, e.g. zeta
Arrays begin with upper case, e.g. Zeta_i
2 D Matrices are all upper case, e.g. AW, ZETA=[Zeta_i]
3 D arrays (tensors) will be labelled with a 3 in the name, e.g. A3

'''

import numpy as np
import sympy as sm
from IPython import embed

#import sympy.tensor as smtens
import sympy.tensor.array as smarr
import linfunc


# 1,2 local index in the element. Node 1 is "ahead"
Zeta01,Zeta02=sm.symbols('Zeta01 Zeta02', real=True)
zeta01_x,zeta02_x,zeta01_y,zeta02_y=\
                   sm.symbols('zeta01_x,zeta02_x,zeta01_y,zeta02_y', real=True)
Kskew2D=smarr.MutableDenseNDimArray([[0,-1],[1,0]])

# Position vectors
Zeta01=smarr.MutableDenseNDimArray([zeta01_x,zeta01_y])
Zeta02=smarr.MutableDenseNDimArray([zeta02_x,zeta02_y])

# define order of derivation
ZetaAllList=[zeta01_x,zeta02_x,zeta01_y,zeta02_y]

# Delta vector (and derivative)
Dzeta=Zeta02-Zeta01
Area=sm.sqrt(linfunc.scalar_product(Dzeta,Dzeta))
Dzeta_unit=Dzeta/Area

# Normal
Norm=linfunc.matrix_product(Kskew2D,Dzeta_unit)
Norm=linfunc.simplify(Norm)

### check norm
assert linfunc.scalar_product(Norm,Dzeta).simplify()==0, 'Norm is wrong'
assert linfunc.scalar_product(Norm,Dzeta_unit).simplify()==0, 'Norm is wrong'
assert linfunc.scalar_product(Norm,Norm).simplify()==1, 'Normal is not unit length'


# Normal by Area derivative
NA=Area*Norm
derNA=sm.derive_by_array(NA,ZetaAllList).transpose()
derNA=linfunc.simplify(derNA)

# Verify Normal by Area derivative
derDzeta=sm.derive_by_array(Dzeta,ZetaAllList).transpose()
derDzeta=linfunc.simplify(derDzeta)

derNA2=linfunc.matrix_product(Kskew2D,derDzeta)
derNA2=linfunc.simplify(derNA2)

check=linfunc.simplify(derNA-derNA2)
assert linfunc.scalar_product(check,check)==0,\
                                   'Normal by Aera derivatives are not the same'


embed()

if __name__=='__main__':
	pass




