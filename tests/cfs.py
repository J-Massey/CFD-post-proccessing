
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: profiles from DNS
@contact: jmom1n15@soton.ac.uk
"""

# Imports
import numpy as np
import postproc
from postproc import cylinder_forces as cf
from postproc import io
from postproc import plotter

D = 32 # Diameter (Characteristic length)
U = 1  # Characteristic velocity
file1 = '39e3_32/fort.9'

# t, fx, fy = io.unpack_flex_forces(D, file)
print(io.unpack_profiles('./39e3_32/profiles/',256,3,theta=p.pi/32))

# print(np.sort(os.listdir('./39e3_32/profiles/')))
# t=t[t/D>=200]
# fy = fy[-t.shape[0]:]
# plotter.plotCL(fy, t/D, 'CL-t.pdf', St=cf.find_St(t,fy,D,U), CL_rms=cf.rms(fy), CD_rms=cf.rms(fx), n_periods=cf.find_num_periods(fy))

# cd /home/masseyjmo/Workspace/Lotus/projects/cf_lotus/ml/real_profiles/DNS/3D/
