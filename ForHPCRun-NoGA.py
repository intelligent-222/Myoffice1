#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import eppy as ep
from eppy import modeleditor
import sys
from eppy.modeleditor import IDF
import pandas as pd
from statistics import mean
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
#import timeit
#import time
#import multiprocessing
from eppy.pytest_helpers import do_integration_tests
from eppy.runner.run_functions import install_paths, EnergyPlusRunError
from eppy.runner.run_functions import multirunner
from eppy.runner.run_functions import run
from eppy.runner.run_functions import runIDFs
#import zeppy
#from zeppy import ppipes
Ncores=int(sys.argv[1])

# In[2]:


path ="/speed-scratch/z_khoras"


# In[3]:


iddfile = path+ '/EP-8-9/EnergyPlus-8-9-0/Energy+.idd'
IDF.setiddname(iddfile)


# In[6]:


"""multiprocessing runs"""

# using generators instead of a list
# when you are running a 100 files you have to use generators

import os 
from eppy.modeleditor import IDF
from eppy.runner.run_functions import runIDFs

def make_options(idf):
    idfversion = idf.idfobjects['version'][0].Version_Identifier.split('.')
    idfversion.extend([0] * (3 - len(idfversion)))
    idfversionstr = '-'.join([str(item) for item in idfversion])
    fname = idf.idfname
    options = {
        'ep_version':idfversionstr,
        'output_prefix':os.path.basename(fname).split('.')[0],
        'output_suffix':'C',
        'output_directory':os.path.dirname(fname),
        'readvars':True,
        'expandobjects':True
        }
    return options




def main(X):
    from eppy.modeleditor import IDF
    iddfile = path+ '/EP-8-9/EnergyPlus-8-9-0/Energy+.idd'
    IDF.setiddname(iddfile)
    epwfile = path+'/EP-INPUTS/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw'
    #w = X
    # w = X.item()
    #w = np.asscalar(X)
    a = X[0]

    z=np.arange(0,301,20)
    b = z[int(X[1])]
    
    t = np.arange(0.3,0.81,0.05)
    c=t[int(X[2])]
  
    j = np.arange(0,1.1,0.1)
    d = j[int(X[3])]

    m = np.arange(0,1.1,0.1)
    e = m[int(X[4])]

    y = [0,90,180,270]
    f = y[int(X[5])]

    n = np.arange(0.05,0.21,0.05)
    g = n[int(X[6])]

    q = np.arange(0.2,0.71,0.05)
    h = q[int(X[7])]



    fname1 = path +'/Final8Var/M1Final3OCCSOV.idf'
    epwfile = path+'/Final8Var/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw'
    idf = IDF(fname1,epwfile)
    occlightingmodel = idf.idfobjects['EnergyManagementsystem:program'][1]
    occlightingmodel.Program_Line_7 ="set x=" + str(a) #maybe we can use directly "set x=" + str(X[0])
   
    occlightingmodel.Program_Line_3 = "set luxmean=" + str(b)

    Windowmaterial = idf.idfobjects['WindowMAterial:SimpleGlazingSystem'][0]
    Windowmaterial.Visible_Transmittance = c

    F16_acoustic_tile=idf.idfobjects['Material'][0]
    F16_acoustic_tile.Visible_Absorptance = d

    G01a_19mm_gypsum_board=idf.idfobjects['Material'][1]
    G01a_19mm_gypsum_board.Visible_Absorptance = e

    office=idf.idfobjects['Building'][0]
    office.North_Axis = f

    Blind_material=idf.idfobjects['WindowMaterial:Shade'][0]
    Blind_material.Visible_Reflectance = g

    idf.saveas(path +'/Final8Var/M1Final3OCCSOV.idf')
    
    from geomeppy import IDF
    fname2 = path +'/Final8Var/M1Final3OCCSOV.idf'
    idf1 = IDF(fname2,epwfile)
    idf1.set_wwr(wwr=0, wwr_map={180:h}, force=True, construction='double pane windows')
    idf1.saveas(path +'/Final8Var/M1Fianl3OCCSOV.idf')


    from eppy.modeleditor import IDF
    fname1 = path +'/Final8Var/M1Final3OCCSOV.idf'
    
    idf = IDF(fname1,epwfile)
    sub_surface=idf.idfobjects['FenestrationSurface:Detailed'][0]
    sub_surface.Shading_Control_Name = 'wshCTRL1'
    idf.saveas(path +'/Final8Var/M1Final3OCCSOV.idf')


    fnames=[]
    for i in range (1,111):
        fname1 = path +'/Final8Var/M1Final3OCCSOV.idf'
        idf = IDF(fname1,epwfile)
        idf.saveas(path +'/Final8Var/M%dFinal3OCCSOV.idf'%(i))
        
        fnames.append(path +'/Final8Var/M%dFinal3OCCSOV.idf'%(i))
        #files = os.listdir(path)
        #fnames = [f for f in files if f[-4:] == '.idf']
    from eppy.modeleditor import IDF
    from eppy.runner.run_functions import runIDFs
    idfs = (IDF(fname, epwfile) for fname in fnames)
    runs = ((idf, make_options(idf) ) for idf in idfs)
    num_CPUs = Ncores
   # num_CPUs = 16
    runIDFs(runs, num_CPUs)







    TRELC=[]

    for i in range (1,111):
        Data=pd.read_csv(path +'/Final8Var/M%dFinal3OCCSOV.csv'%(i))
        ELC=Data['LIGHT:Lights Electric Energy [J](TimeStep)'].sum()*2.78*10**(-7)
        TRELC.append(ELC)
    return np.sum(TRELC)

#import time
#starttime = time.time()
#if __name__ == '__main__':
    
    
#    main()
#print('Time taken = {} seconds'.format(time.time() - starttime))
    


# In[7]:


#print(main(3))


# In[ ]:
algorithm_param = {'max_num_iteration':50,'population_size':15,'mutation_probability':0.2,'elit_ratio':0.01,'crossover_probability':0.7,'parents_portion':0.3,'crossover_type':'uniform','max_iteration_without_improv':20}


varbound = np.array([[1,120],[0,15],[0,10],[0,10],[0,10],[0,3],[0,3],[0,10]])
vartype = np.array([['int'],['int'],['int'],['int'],['int'],['int'],['int'],['int']])
model = ga(function=main, dimension=8, variable_type_mixed=vartype, variable_boundaries = varbound,function_timeout = 20000,algorithm_parameters = algorithm_param)

model.run()



