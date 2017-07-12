'''
Created on 21 Sep 2015

@author: sm6110
'''

import h5py
import os
from numpy import ndarray, float32


def h5file(savedir,h5filename, *class_inst ):
    '''
    Creates h5filename and saves all the classes specified after the
    first input argument
    
    @param savedir: target directory
    @param h5filename: file name
    @param *class_inst: a number of classes to save 
    '''

    os.system('mkdir -p %s'%savedir)
    h5filename=os.path.join(savedir,h5filename)
    
    hdfile=h5py.File(h5filename,'w')
    
    for cc in class_inst:
        add_class_as_grp(cc,hdfile)

    hdfile.close()
    return None


def add_class_as_grp(obj,hdfile,compress=False):
    '''
    Given a class instance 'obj', the routine adds it as a group to a hdf5 file
    
    Remark: the previous content of the file is not deleted or modified. If the
    class already exists, an error occurs. 
    
    Warning: multiple calls of this function will lead to a file increase
    
    If compress is True, numpy arrays will be saved in single precisions.
    Not float numbers.
    
    '''
    
    # look for a name, otherwise use class name
    if hasattr(obj,'name'): grp = hdfile.create_group(obj.name)   
    else: grp = hdfile.create_group(obj.__class__.__name__)
    
    # extract all attributes
    dictname=obj.__dict__
    
    for attr in dictname:
        value=getattr(obj,attr)
        
        if value != None:
            # check for c_types
            if type(value).__name__[:2]=='c_': value=value.value
            try:
                if compress is True and type(value) is ndarray:
                    grp[attr]=float32(value)
                else:       
                    grp[attr]=value
            except TypeError:
                #print ('TypeError occurred when saving %s' %attr)
                grp[attr]='TypeError'
            except:
                #print ('unknown error occurred when saving %s' %attr)
                grp[attr]='UnknownError'

    return hdfile   


def all_class(obj,filename):
    '''
    Given a class instance 'obj', the function saves all its attributes in a
    subgroup of an h5 file 'filename'.
    
    Note: the method can be called multiple times to save input/output of
    simulation.
    
    Remark: the previous content of the file is not deleted except when the 
    class already exists as a group in the file: in this case all its attributes 
    will e overwritten.
    
    Warning: - multiple calls to this function will lead to a file increase. 
             - FIX OR USE add_class_as_grp
    
    '''

    #read/write - create otherwise
    hdfile = h5py.File(filename,'a') 

    try: # to read
        grp = hdfile[obj.__class__.__name__]
    except KeyError: # create group
        grp = hdfile.create_group(obj.__class__.__name__)
    
    # extract all attributes
    dict=obj.__dict__
    
    for attr in dict:
        value=getattr(obj,attr)
        
        # check is not empty...
        if value != None:
            
            if type(value).__name__[:2]=='c_':
                print('%s is a ctype:'%attr)
                value=value.value
            
            try:
                grp[attr]=value
            except RuntimeError: # attribute exists
                try:
                    print('Modify value',value)
                    hdfile.attrs.modify(attr,value)
                except TypeError:
                    print ('TypeError occurred when saving', value)
            except:
                print ('unknown error occurred when saving %s' %attr)
                #grp[attr]='unknown error occurred' 
              
        #else:     
            #except TypeError: # empty field
            #    grp[attr]='empty'  
        #    grp[attr]='empty'


    hdfile.close()  
    
    return None   



if __name__=='__main__':
        
    import os
    import sys
    sys.path.append( os.environ["SHARPYDIR"]+'/src' )
    sys.path.append( os.environ["SHARPYDIR"]+'/src/Main' )   
    import SharPySettings 
    
    import DerivedTypes
    import DerivedTypesAero
    from PyCoupled.Utils import DerivedTypesAeroelastic
    
    import numpy as np
    import ctypes as ct
    
    XBOPTS = DerivedTypes.Xbopts(FollowerForce = ct.c_bool(False),\
                                         MaxIterations = ct.c_int(50),\
                                         PrintInfo = ct.c_bool(True),\
                                         NumLoadSteps = ct.c_int(25),\
                                         Solution = ct.c_int(312),\
                                         MinDelta = ct.c_double(1e-5),\
                                         NewmarkDamp = ct.c_double(1.0e-4))
                    
    XBINPUT = DerivedTypes.Xbinput(3,10)
    XBINPUT.BeamLength = 6.096
    XBINPUT.BeamStiffness[0,0] = 1.0e+09
    XBINPUT.BeamStiffness[1,1] = 1.0e+09
    XBINPUT.BeamStiffness[2,2] = 1.0e+09
    XBINPUT.BeamStiffness[3,3] = 0.9875e+06
    XBINPUT.BeamStiffness[4,4] = 9.77e+06
    XBINPUT.BeamStiffness[5,5] = 9.77e+08
                
    # Get suggested panelling.
    Umag = 140.0
    M = 16
    c = 1.8288
    delTime = c/(Umag*M)
    
    # Unsteady parameters.
    XBINPUT.dt = delTime
    XBINPUT.t0 = 0.0
    XBINPUT.tfin = 0.5
     
    # aero params.
    WakeLength = 30.0*c
    Mstar = int(WakeLength/(delTime*Umag))
           
    VMOPTS = DerivedTypesAero.VMopts(M = M,
                                     N = XBINPUT.NumNodesTot-1,
                                     ImageMethod = True,
                                     Mstar = Mstar,
                                     Steady = False,
                                     KJMeth = True,
                                     NewAIC = True,
                                     DelTime = delTime,
                                     NumCores = 4)
                                
    "aero inputs"
    VMINPUT = DerivedTypesAero.VMinput(c = c, b = XBINPUT.BeamLength,\
                                       U_mag = 20.0,\
                                       alpha = 2.0*np.pi/180.0,\
                                       theta = 0.0,\
                                       WakeLength = WakeLength)         
            
    AELAOPTS = DerivedTypesAeroelastic.AeroelasticOps(ElasticAxis = -0.34 ,
                                                      InertialAxis = -7.0/50.0,
                                                      AirDensity = 1.02,
                                                      Tight = False,
                                                      ImpStart = False) 
    
    #### method 1
    print('1')
    all_class(XBINPUT,'testfile.h5')
    print('2')
    all_class(XBOPTS,'testfile.h5')
    print('3')
    all_class(VMOPTS,'testfile.h5')
    print('4')
    all_class(VMINPUT,'testfile.h5')
    print('5')
    all_class(AELAOPTS,'testfile.h5')
    
    
    ### method 2
    hdfile=h5py.File('testfile02.h5','w')
    add_class_as_grp(AELAOPTS,hdfile)
    add_class_as_grp(VMINPUT,hdfile)
    add_class_as_grp(VMOPTS,hdfile)
    add_class_as_grp(XBOPTS,hdfile)
    add_class_as_grp(XBINPUT,hdfile)
    

      



                

                    