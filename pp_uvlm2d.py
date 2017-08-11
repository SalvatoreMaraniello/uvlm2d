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

    scale_aero=.5*np.linalg.norm(S.Zeta[0,:]-S.Zeta[-1,:])/np.max(np.abs(S.Faero))
    K=S.K

    # create a continuous contour
    Coord=np.zeros((2*K+1,2))
    Coord[:K,0]=S.Zeta[:,0]
    Coord[:K,1]=S.Zeta[:,1]
    Coord[K:-1,0]=S.Zeta[:,0][::-1]+scale_aero*S.Faero[::-1,0]
    Coord[K:-1,1]=S.Zeta[:,1][::-1]+scale_aero*S.Faero[::-1,1]
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
    ax.plot(S.Zeta_c[:,0],S.Zeta_c[:,1],'r',marker='o',linestyle='',
                                                label=r'collocation points')
    ax.legend()

    return ax, fig
