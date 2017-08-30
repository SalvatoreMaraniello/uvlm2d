'''
Created on 21 Sep 2015

@author: sm6110
'''

import h5py
import os
from numpy import ndarray, float32
from IPython import embed


def h5file(savedir,h5filename, *class_inst):
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
        #print('Adding %s' %cc)
        add_class_as_grp(cc,hdfile)


    hdfile.close()
    return None


def add_class_as_grp(obj,grpParent,compress=False):
    '''
    Given a class instance 'obj', the routine adds it as a group to a hdf5 file
    
    Remark: the previous content of the file is not deleted or modified. If the
    class already exists, an error occurs. 
    
    Warning: multiple calls of this function will lead to a file increase
    
    If compress is True, numpy arrays will be saved in single precisions.
    Not float numbers.
    
    '''
    
    # look for a name, otherwise use class name
    if hasattr(obj,'name'): 
        grp = grpParent.create_group(obj.name)   
    else: 
        grp = grpParent.create_group(obj.__class__.__name__)
    
    # extract all attributes
    dictname=obj.__dict__

    for attr in dictname:
        value=getattr(obj,attr)
        if value is not None:
            # Add Output class as subgroup
            if isinstance(value,Output):
                add_class_as_grp(value,grp,compress=compress)
            # Add c_types
            if type(value).__name__[:2]=='c_': value=value.value
            # Add numpy arrays
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

    return grpParent   


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
        if value is not None:  
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



class Output:
    '''
    Class to store output
    '''
    
    def __init__(self,name=None):
        self.name=name
        
    def drop(self, **kwargs):
        '''Attach random variables to this class'''
        for ww in kwargs:
            setattr(self, ww, kwargs[ww])
        
        return self

                 