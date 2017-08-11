'''
Analytical linearisation of Joukovski force w.r.t. changes of induced velocity
due to grid perturbations only, i.e.
	dFjouk = pder{Fjouk,uind} dvind =  C pder{uind,zeta}
where C is a constant term.

This derivatives are taken w.r.t.:
- the nodes of vortex ring v
- the nodes of the segment s at which the induced velocity is computed

When computing the induced velocity at the segment, only the net vorticity is
used. Therefore, this function returns the derivative of the net induced velocity
induced over segment ZetaC by segment Zeta01,

Sign convention:
Scalar quantities are all lower case, e.g. zeta
Arrays begin with upper case, e.g. Zeta_i
2 D Matrices are all upper case, e.g. AW, ZETA=[Zeta_i]
3 D arrays (tensors) will be labelled with a 3 in the name, e.g. A3
'''

import numpy as np
import sympy as sm
import sympy.tensor.array as smarr
import linfunc


### Position vectors of vortex element
Zeta01=sm.symbols('Zeta01', real=True)
zeta01_x,zeta01_y=sm.symbols('zeta01_x,zeta01_y', real=True)
Zeta01=smarr.MutableDenseNDimArray([zeta01_x,zeta01_y])

### Segment point
ZetaC=sm.symbols('ZetaC', real=True)
zetaC_x,zetaC_y=sm.symbols('zetaC_x,zetaC_y',real=True)
ZetaC=smarr.MutableDenseNDimArray([zetaC_x,zetaC_y])


### Other symbles/constants
Kskew2D=smarr.MutableDenseNDimArray([[0,-1],[1,0]])
CF,gamma01,gammaC=sm.symbols('CF gamma01 gammaC', real=True)


# define order of derivation
ZetaAllList=[zeta01_x,zeta01_y, # w.r.t. segment 01
			 zetaC_x,zetaC_y    # w.r.t. segment C
			]
# Delta vector (and derivative)
R01=ZetaC-Zeta01

# Velocity induced by a segment
def Vind_segment(R,cf,g):
	BiotTerm=cf*g* R/linfunc.scalar_product(R,R)
	Qind=linfunc.matrix_product(Kskew2D,BiotTerm)
	return linfunc.simplify(Qind)

# Joukovski force of secment C from segment 01
Fjouk=-gammaC*linfunc.matrix_product(Kskew2D,Vind_segment(R01,CF,gamma01))
Fjouk=linfunc.simplify(Fjouk)

### Derivative
Der=sm.derive_by_array(Fjouk,ZetaAllList).transpose() # 2x6
#Der=linfunc.simplify(Der)

### Verification
BiotTerm_vortex = +CF*gamma01*R01/linfunc.scalar_product(R01,R01)
derBiotTerm=sm.derive_by_array(BiotTerm_vortex,ZetaAllList).transpose()
derQind=linfunc.matrix_product(Kskew2D,derBiotTerm)
Der2=-gammaC*linfunc.matrix_product(Kskew2D,derQind)

check=linfunc.simplify(Der-Der2)
assert linfunc.scalar_product(check,check)==0,'Derivatives are not the same'

################ we use Der2 as it has a shorter form ##########################
Der=Der2


### Shorten expressions
R01_x,R01_y=sm.symbols('R01_x,R01_y', real=True)

Der=linfunc.subs(Der, -zeta01_x + zetaC_x , R01_x )
Der=linfunc.subs(Der, -zeta01_y + zetaC_y , R01_y )

Der=linfunc.subs(Der, -2*zeta01_x + 2*zetaC_x, 2*R01_x )
Der=linfunc.subs(Der, -2*zeta01_y + 2*zetaC_y, 2*R01_y )

R01sq=sm.symbols('R01sq', real=True)
Der=linfunc.subs(Der,R01_x**2 + R01_y**2,R01sq)

print('Final simplification...')
Der=linfunc.simplify(Der)
for dd in Der:
	print('\t\t%s ,\n' %dd)


1/0
del Der,Der2

print('Derivatives w.r.t. segment 1 only - last wake vortex does not include segment 2')

# Full vortex velocity
Qind_vortex_Mw = Vind_segment(R01,CF,Gamma)
Fjouk_Mw=-gamma_s*linfunc.matrix_product(Kskew2D,Qind_vortex_Mw)

# Derivative
Der_Mw=sm.derive_by_array(Fjouk_Mw,ZetaAllList).transpose()
Der_Mw=linfunc.simplify(Der_Mw)


### Verification
BiotTerm_vortex_Mw = CF*Gamma*R01/linfunc.scalar_product(R01,R01)
derBiotTerm_Mw=sm.derive_by_array(BiotTerm_vortex_Mw,ZetaAllList).transpose()
derQind_Mw=linfunc.matrix_product(Kskew2D,derBiotTerm_Mw)

Der2_Mw=-gamma_s*linfunc.matrix_product(Kskew2D,derQind_Mw)

check_Mw=linfunc.simplify(Der_Mw-Der2_Mw)
assert linfunc.scalar_product(check_Mw,check_Mw)==0,'Derivatives are not the same'


### Shorten expressions

Der_Mw=linfunc.subs(Der_Mw, -zeta01_x + zetaC_x , R01_x )
Der_Mw=linfunc.subs(Der_Mw, -zeta01_y + zetaC_y , R01_y )

Der_Mw=linfunc.subs(Der_Mw, -2*zeta01_x + 2*zetaC_x, 2*R01_x )
Der_Mw=linfunc.subs(Der_Mw, -2*zeta01_y + 2*zetaC_y, 2*R01_y )

Der_Mw=linfunc.subs(Der_Mw,R01_x**2 + R01_y**2,R01sq)

print('Final simplification...')
Der_Mw=linfunc.simplify(Der_Mw)

for dd in Der_Mw:
	print('\t\t%s ,\n' %dd)

