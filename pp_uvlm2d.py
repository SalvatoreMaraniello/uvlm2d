'''
Tools to visualise results
'''


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
patches = []

from IPython import embed 


def force_distr(S,figname='Force distribution'):
    '''
    Visualise force distribution along aerofoil
    '''

    caero='b'
    cchord='k'

    scale_aero=.5*np.linalg.norm(S.Zeta[0,:]-S.Zeta[-1,:])/np.max(np.abs(S.FmatSta))
    M=S.M

    # create a continuous contour
    Coord=np.zeros((2*M+1,2))
    Coord[:M,0]=S.Zeta[:M,0]
    Coord[:M,1]=S.Zeta[:M,1]
    Coord[M:-1,0]=S.Zeta[:M,0][::-1]+scale_aero*S.FmatSta[::-1,0]
    Coord[M:-1,1]=S.Zeta[:M,1][::-1]+scale_aero*S.FmatSta[::-1,1]
    Coord[-1,0]=S.Zeta[0,0]
    Coord[-1,1]=S.Zeta[0,1]

    ### start plot
    C_patches=[]
    fig = plt.figure(figname,(10,6))
    ax = fig.add_subplot(111)
    # add patch
    polygon=Polygon(Coord,closed=True,edgecolor=caero,facecolor=caero,alpha=0.3)
    ax.add_patch(polygon)
    # and underline aero load
    ax.plot(Coord[:,0],Coord[:,1],color=caero,lw=1)
    # plot chord section
    ax.plot(S.Zeta[:,0],S.Zeta[:,1],cchord,lw=3,label=r'chord')

    return ax, fig



def visualise_grid(S,figname='Geometry and grid'):
    '''
    Visualise aerofoil and wake grid/collocation points.
    The input S is a class uvlm2d.solver
    '''

    fig=plt.figure(figname, figsize=[10.,6.0])
    ax=fig.add_subplot(111)
    #ax.plot(S.Rmat[:,0],S.Rmat[:,1],'0.6',marker='s',lw=3,
    #                                                    label=r'wing panel')
    ax.plot(S.Zeta[:,0],S.Zeta[:,1],'k',marker='x',
                                                     label=r'wing vortices')
    ax.plot(S.ZetaW[:,0],S.ZetaW[:,1],'k',marker='*',
                                                     label=r'wake vortices')
    ax.plot(S.Cmat[:,0],S.Cmat[:,1],'r',marker='o',linestyle='',
                                                label=r'collocation points')
    ax.legend()

    return ax, fig







'''

# deformed wing front view

fig_ads_comp = plt.figure('Aero Distribution Comparison', figsize=[10.,6.0])
ax = fig_ads_comp.add_subplot(111)

for ii in ListComp:#range(Nsig):

    Coord=np.zeros((2*Kzeta+1,2))

    # beam deformed
    xcoord = THPosDefGlobal[ii][-1,:,0]
    zcoord = THPosDefGlobal[ii][-1,:,2]
    Coord[:Kzeta,0]=xcoord
    Coord[:Kzeta,1]=zcoord
    Coord[Kzeta:-1,0]=xcoord[::-1]+scale_aero*FadGList[ii,-1,::-1,0]
    Coord[Kzeta:-1,1]=zcoord[::-1]+scale_aero*FadGList[ii,-1,::-1,2]
    Coord[-1,0]=xcoord[0]
    Coord[-1,1]=zcoord[0]
    
    # add patch
    polygon = Polygon(Coord, closed=True, edgecolor=cscheme[ii], facecolor=cscheme[ii], alpha=0.3)
    ax.add_patch(polygon)

    # add think line for wing
    ax.plot( xcoord, zcoord, color=cscheme[ii], linewidth=3  )
    # and aero load
    ax.plot( Coord[:,0], Coord[:,1], color=cscheme[ii], linewidth=1  )
    
    # hinge
    #ax.plot( [0], [0], '.4', marker='o', markersize=10 )
    
    # add patch for artistic legend
    C_patches.append( mpl.lines.Line2D( [], [],color=cscheme[ii], linewidth=2,  
                                        linestyle='-', label=sigleg[ii] )  )


if SmallPlot:
    ax.set_xlim(-17.0,17.0)
    ax.set_ylim(-7.0,15.0)
elif 'sig11' in h5filename:
    ax.set_xlim(-17.0,12.0)
    ax.set_ylim(-2.0,15.0)
elif 'sig15' in h5filename:
    ax.set_xlim(-17.0,13.0)
    ax.set_ylim(-3.0,14.0)    
elif 'sig50' in h5filename:
    ax.set_xlim(-17.0,17.0)
    ax.set_ylim(-7.0,11.0)  
 
    
    
ax.set_xlabel(r'[m]',fontsize=fontlabel)
ax.set_ylabel(r'[m]',fontsize=fontlabel)
ax.set_aspect('equal')

cleg=ax.legend(handles=C_patches,loc='upper left')
fig_ads_comp.gca().add_artist(cleg)                                        

plt.savefig(savecode+'aero_on_deform_wing.pdf')
plt.savefig(savecode+'aero_on_deform_wing.png')

'''