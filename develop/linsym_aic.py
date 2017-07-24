'''
Analytical linearisation of Wnc0 * A * Gamma matrix. This corresponds to the
analytical derivative of Vind(n_c=const,c,v), normal velocity induced by vortex
v over the collocation point c, assuming the normal to be constant.

This derivatives are taken w.r.t.:
- the nodes of vortex ring v
- the nodes of the vortex ring c
Note that these points can coincide.

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
Zeta01,Zeta02=sm.symbols('Zeta01 Zeta02', real=True)
zeta01_x,zeta02_x,zeta01_y,zeta02_y=\
                   sm.symbols('zeta01_x,zeta02_x,zeta01_y,zeta02_y', real=True)
Zeta01=smarr.MutableDenseNDimArray([zeta01_x,zeta01_y])
Zeta02=smarr.MutableDenseNDimArray([zeta02_x,zeta02_y])



### Position vertices of ring with collocation point
ZetaA,ZetaB=sm.symbols('ZetaA ZetaB', real=True)
zetaA_x,zetaB_x,zetaA_y,zetaB_y=\
                   sm.symbols('zetaA_x,zetaB_x,zetaA_y,zetaB_y', real=True)
ZetaA=smarr.MutableDenseNDimArray([zetaA_x,zetaA_y])
ZetaB=smarr.MutableDenseNDimArray([zetaB_x,zetaB_y])


### Collocation point
ZetaC,ZetaDummy=sm.symbols('ZetaC ZetaDummy', real=True)
zetaC_x,zetaC_y,zetaDummy_x,zetaDummy_y=sm.symbols(
	        'zetaC_x,zetaC_y,zetaDummy_x,zetaDummy_y',real=True)
ZetaC=smarr.MutableDenseNDimArray([zetaC_x,zetaC_y])
#ZetaDummy=smarr.MutableDenseNDimArray([zetaDummy_x,zetaDummy_y])


### Define collocation point position
ZetaC=ZetaA/2+ZetaB/2


# ### Define collocation point equation
# if   case=='inner': ZetaC=Zeta01/2+Zeta02/2
# elif case=='nfwd' : ZetaC=Zeta01/2+ZetaDummy/2	
# elif case=='nback':	ZetaC=Zeta02/2+ZetaDummy/2	
# elif case=='outer': pass
# else: raise NameError('Wrong case!')	


### Normal at collocation point
N0=sm.symbols('N0', real=True)
n0_x,n0_y=sm.symbols('n0_x,n0_y', real=True)
N0=smarr.MutableDenseNDimArray([n0_x,n0_y])

### Other symbles/constants
Kskew2D=smarr.MutableDenseNDimArray([[0,-1],[1,0]])
CF,gamma=sm.symbols('CF gamma', real=True)


# define order of derivation
ZetaAllList=[zeta01_x,zeta02_x,zeta01_y,zeta02_y, # w.r.t. panel nodes
			 zetaA_x,zetaB_x,zetaA_y,zetaB_y, # w.r.t. collocation point ring
			]


# Delta vector (and derivative)
R01=ZetaC-Zeta01
R02=ZetaC-Zeta02


# Velocity induced by a segment
def Vind_perp_segment(R,cf,g,nv):
	BiotTerm=cf*g* R/linfunc.scalar_product(R,R)
	Qind=linfunc.matrix_product(Kskew2D,BiotTerm)
	Qind_perp=linfunc.scalar_product(nv,Qind)
	return Qind_perp.simplify()

# Full vortex velocity
Qind_perp_vortex = + Vind_perp_segment(R01,CF,gamma,N0) \
				   - Vind_perp_segment(R02,CF,gamma,N0)

# Derivative
Der=linfunc.scalar_deriv(Qind_perp_vortex,ZetaAllList)#.transpose()
#Der=linfunc.simplify(Der)

### Verification
BiotTerm_vortex = +CF*gamma*R01/linfunc.scalar_product(R01,R01) \
			      -CF*gamma*R02/linfunc.scalar_product(R02,R02) 
derBiotTerm=sm.derive_by_array(BiotTerm_vortex,ZetaAllList)
derQind=linfunc.matrix_product(Kskew2D,derBiotTerm.transpose())
derQind_perp=linfunc.matrix_product(N0.reshape(1,2),derQind)
Der2=derQind_perp.reshape(8,)
#Der2=linfunc.simplify(Der2)


check=linfunc.simplify(Der-Der2)
assert linfunc.scalar_product(check,check)==0,'Derivatives are not the same'


### Shorten expressions
R01_x,R01_y=sm.symbols('R01_x,R01_y', real=True)
R02_x,R02_y=sm.symbols('R02_x,R02_y', real=True)
Der=linfunc.subs(Der,-2*zeta01_x+zetaA_x+zetaB_x,2*R01_x)
Der=linfunc.subs(Der,-2*zeta02_x+zetaA_x+zetaB_x,2*R02_x)
Der=linfunc.subs(Der,-2*zeta01_y+zetaA_y+zetaB_y,2*R01_y)
Der=linfunc.subs(Der,-2*zeta02_y+zetaA_y+zetaB_y,2*R02_y)

R01sq,R02sq=sm.symbols('R01sq,R02sq', real=True)
Der=linfunc.subs(Der,R01_x**2 + R01_y**2,R01sq)
Der=linfunc.subs(Der,R02_x**2 + R02_y**2,R02sq)


print('Final simplification...')
Der=linfunc.simplify(Der)


for dd in Der:
	print('\t\t%s ,\n' %dd)




print('Derivatives w.r.t. segment 1 only - last wake vortex does not include segment 2')

# Full vortex velocity
Qind_perp_vortex_Mw = + Vind_perp_segment(R01,CF,gamma,N0)

# Derivative
Der_Mw=linfunc.scalar_deriv(Qind_perp_vortex_Mw,ZetaAllList)#.transpose()
#Der=linfunc.simplify(Der)

### Verification
BiotTerm_vortex_Mw = +CF*gamma*R01/linfunc.scalar_product(R01,R01)
derBiotTerm_Mw=sm.derive_by_array(BiotTerm_vortex_Mw,ZetaAllList)
derQind_Mw=linfunc.matrix_product(Kskew2D,derBiotTerm_Mw.transpose())
derQind_perp_Mw=linfunc.matrix_product(N0.reshape(1,2),derQind_Mw)
Der2_Mw=derQind_perp_Mw.reshape(8,)
#Der2=linfunc.simplify(Der2)


check_Mw=linfunc.simplify(Der_Mw-Der2_Mw)
assert linfunc.scalar_product(check_Mw,check_Mw)==0,'Derivatives are not the same'

### Shorten expressions
Der_Mw=linfunc.subs(Der_Mw,-2*zeta01_x+zetaA_x+zetaB_x,2*R01_x)
Der_Mw=linfunc.subs(Der_Mw,-2*zeta01_y+zetaA_y+zetaB_y,2*R01_y)
Der_Mw=linfunc.subs(Der_Mw,R01_x**2 + R01_y**2,R01sq)

print('Final simplification...')
Der_Mw=linfunc.simplify(Der_Mw)

for dd in Der_Mw:
	print('\t\t%s ,\n' %dd)

