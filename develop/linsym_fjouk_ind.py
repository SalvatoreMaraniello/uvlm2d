'''
Analytical linearisation of Joukovski force w.r.t. changes of induced velocity
due to grid perturbations only, i.e.
	dFjouk = pder{Fjouk,uind} dvind =  C pder{uind,zeta}
where C is a constant term.

This derivatives are taken w.r.t.:
- the nodes of vortex ring v
- the nodes of the segment s at which the induced velocity is computed

Two copy of the segment s vertices, ZetaCA and ZetaCB are created. While these
points are the same, this separation allows us to neglect the contribution
of the segment01 of the vortex when Zeta01=ZetaCA and so on.

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


### Segment point
ZetaCA,ZetaCB=sm.symbols('ZetaCA,ZetaCB', real=True)
zetaCA_x,zetaCA_y,zetaCB_x,zetaCB_y=sm.symbols(
	                            'zetaCA_x,zetaCA_y,zetaCB_x,zetaCB_y',real=True)
ZetaCA=smarr.MutableDenseNDimArray([zetaCA_x,zetaCA_y])
ZetaCB=smarr.MutableDenseNDimArray([zetaCB_x,zetaCB_y])

### Other symbles/constants
Kskew2D=smarr.MutableDenseNDimArray([[0,-1],[1,0]])
CF,Gamma,gamma_s=sm.symbols('CF Gamma gamma_s', real=True)

# define order of derivation
ZetaAllList=[zeta01_x,zeta02_x,zeta01_y,zeta02_y, # w.r.t. panel nodes
			 zetaCA_x,zetaCB_x,zetaCA_y,zetaCB_y  # w.r.t. segment vertex
			]

# Delta vector (and derivative)
R01=ZetaCA-Zeta01
R02=ZetaCB-Zeta02

# Velocity induced by a segment
def Vind_segment(R,cf,g):
	BiotTerm=cf*g* R/linfunc.scalar_product(R,R)
	Qind=linfunc.matrix_product(Kskew2D,BiotTerm)
	return linfunc.simplify(Qind)

# Full vortex velocity
Qind_vortex = Vind_segment(R01,CF,Gamma) - Vind_segment(R02,CF,Gamma)

# Joukovski force
Fjouk=-gamma_s*linfunc.matrix_product(Kskew2D,Qind_vortex)
Fjouk=linfunc.simplify(Fjouk)

### Derivative
Der=sm.derive_by_array(Fjouk,ZetaAllList).transpose() # 2x6
#Der=linfunc.simplify(Der)

### Verification
BiotTerm_vortex = +CF*Gamma*R01/linfunc.scalar_product(R01,R01) \
			      -CF*Gamma*R02/linfunc.scalar_product(R02,R02) 
derBiotTerm=sm.derive_by_array(BiotTerm_vortex,ZetaAllList).transpose()
derQind=linfunc.matrix_product(Kskew2D,derBiotTerm)
Der2=-gamma_s*linfunc.matrix_product(Kskew2D,derQind)

check=linfunc.simplify(Der-Der2)
assert linfunc.scalar_product(check,check)==0,'Derivatives are not the same'

################ we use Der2 as it has a shorter form ##########################
Der=Der2


### Shorten expressions
R01_x,R01_y=sm.symbols('R01_x,R01_y', real=True)
R02_x,R02_y=sm.symbols('R02_x,R02_y', real=True)

Der=linfunc.subs(Der, -zeta01_x + zetaCA_x , R01_x )
Der=linfunc.subs(Der, -zeta01_y + zetaCA_y , R01_y )
Der=linfunc.subs(Der, -zeta02_x + zetaCB_x , R02_x )
Der=linfunc.subs(Der, -zeta02_y + zetaCB_y , R02_y )

Der=linfunc.subs(Der, -2*zeta01_x + 2*zetaCA_x, 2*R01_x )
Der=linfunc.subs(Der, -2*zeta01_y + 2*zetaCA_y, 2*R01_y )
Der=linfunc.subs(Der, -2*zeta02_x + 2*zetaCB_x, 2*R02_x )
Der=linfunc.subs(Der, -2*zeta02_y + 2*zetaCB_y, 2*R02_y )


R01sq,R02sq=sm.symbols('R01sq,R02sq', real=True)
Der=linfunc.subs(Der,R01_x**2 + R01_y**2,R01sq)
Der=linfunc.subs(Der,R02_x**2 + R02_y**2,R02sq)


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
Der_Mw=linfunc.subs(Der_Mw, -zeta01_x + zetaCA_x , R01_x )
Der_Mw=linfunc.subs(Der_Mw, -zeta01_y + zetaCA_y , R01_y )

Der_Mw=linfunc.subs(Der_Mw, -2*zeta01_x + 2*zetaCA_x, 2*R01_x )
Der_Mw=linfunc.subs(Der_Mw, -2*zeta01_y + 2*zetaCA_y, 2*R01_y )

Der_Mw=linfunc.subs(Der_Mw,R01_x**2 + R01_y**2,R01sq)

print('Final simplification...')
Der_Mw=linfunc.simplify(Der_Mw)

for dd in Der_Mw:
	print('\t\t%s ,\n' %dd)

