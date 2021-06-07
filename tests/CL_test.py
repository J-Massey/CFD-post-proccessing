# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: package tests
@contact: b.fontgarcia@soton.ac.uk
"""

# Imports
import postproc.cylinder_forces as cf
import postproc.io as io
import postproc.plotter as plotter

D = 90 # Diameter (Characteristic length)
U = 1  # Characteristic velocity
file = 'sample_data/CL_3D_zPI.txt'

t, fx, fy = io.unpack3Dforces(file, D)
# torch=torch[torch/length_scale>=200]
# fy = fy[-torch.shape[0]:]
plotter.plotCL(fy, t/D, 'CL-t.svg', St=cf.find_St(t,fy,D,U), CL_rms=cf.rms(fy), CD_rms=cf.rms(fx),
               n_periods=cf.find_num_periods(fy))
