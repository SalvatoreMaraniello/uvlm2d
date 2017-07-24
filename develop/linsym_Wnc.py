'''
Analytical linearisation of Wnc matrix.

Sign convention:

Scalar quantities are all lower case, e.g. zeta
Arrays begin with upper case, e.g. Zeta_i
2 D Matrices are all upper case, e.g. AW, ZETA=[Zeta_i]
3 D arrays (tensors) will be labelled with a 3 in the name, e.g. A3

'''

import numpy as np
import sympy as sm
#import sympy.tensor as smtens
import sympy.tensor.array as smarr
import linfunc


# 1,2 local index in the element. Node 1 is "ahead"
Zeta01,Zeta02=sm.symbols('Zeta01 Zeta02', real=True)
zeta01_x,zeta02_x,zeta01_y,zeta02_y=\
                   sm.symbols('zeta01_x,zeta02_x,zeta01_y,zeta02_y', real=True)
V0_x,V0_y=sm.symbols('V0_x,V0_y', real=True)
V0=smarr.MutableDenseNDimArray([V0_x,V0_y])

Kskew2D=smarr.MutableDenseNDimArray([[0,-1],[1,0]])

# Position vectors
Zeta01=smarr.MutableDenseNDimArray([zeta01_x,zeta01_y])
Zeta02=smarr.MutableDenseNDimArray([zeta02_x,zeta02_y])

# define order of derivation
ZetaAllList=[zeta01_x,zeta02_x,zeta01_y,zeta02_y]

# Delta vector (and derivative)
Dzeta=Zeta02-Zeta01
Dzeta_unit=Dzeta/sm.sqrt(linfunc.scalar_product(Dzeta,Dzeta))
#Dzeta_unit=Dzeta_unit.simplify()

# Normal
Norm=linfunc.matrix_product(Kskew2D,Dzeta_unit)
Norm=linfunc.simplify(Norm)

### check norm
assert linfunc.scalar_product(Norm,Dzeta).simplify()==0, 'Norm is wrong'
assert linfunc.scalar_product(Norm,Dzeta_unit).simplify()==0, 'Norm is wrong'
assert linfunc.scalar_product(Norm,Norm).simplify()==1, 'Normal is not unit length'

# Normal velocity
Vperp=linfunc.scalar_product(Norm,V0)



### Derivatives / with verification
# derX[i,j] is the derivative of X[i] w.r.t ZetaAllList[j]

# Normal
derNorm=sm.derive_by_array(Norm,ZetaAllList).transpose()
derNorm=linfunc.simplify(derNorm)

# Verify Normal derivative
derDzeta_unit=sm.derive_by_array(Dzeta_unit,ZetaAllList).transpose()
derDzeta_unit=linfunc.simplify(derDzeta_unit)

derNorm2=linfunc.matrix_product(Kskew2D,derDzeta_unit)
derNorm2=linfunc.simplify(derNorm2)

check=linfunc.simplify(derNorm-derNorm2)
assert linfunc.scalar_product(check,check)==0,\
                                           'Normal derivatives are not the same'


# Derivatived Norm * V0
derVperp=linfunc.scalar_deriv(Vperp,ZetaAllList) # 1d array
derVperp=linfunc.simplify(derVperp)

# verify...
derVperp2=linfunc.matrix_product(derNorm.transpose(),V0)
check=linfunc.simplify(derVperp-derVperp2)
assert linfunc.scalar_product(check,check)==0,\
                           'Perpendicular velocity derivatives are not the same'


### Shorten expression
dz_x,dz_y=sm.symbols('dz_x,dz_y', real=True)
derVperp=linfunc.subs(derVperp,zeta02_x-zeta01_x,dz_x)
derVperp=linfunc.subs(derVperp,zeta02_y-zeta01_y,dz_y)


if __name__=='__main__':


	pass




