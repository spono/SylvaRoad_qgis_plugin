# -*- coding: utf8 -*-
"""
Software: 0_SylvaRoaD_param
File: SylvaRoaD.py
Copyright (C) Sylvain DUPIRE - SylvaLab 2021
Authors: Sylvain DUPIRE
Contact: sylvain.dupire@sylvalab.fr
Version: 2.2
Date: 2021/08/09
License :  GNU-GPL V3
"""

from SylvaRoaD_1_function import *

###############################################################################
### Parameters
###############################################################################
Wspace= "F:/SylvaLab/Projets/ONF/ONF_Desserte/Work/test_data/Zone test/DATA_STPIERRE/STPIERRE/"

Dtm_file = Wspace+"Input/Raster/MNT_2m_with_nodata.tif"
Obs_Dir =  Wspace+"Input/Obs/"
Waypoints_file = Wspace+"Input/Shape/waypoints.shp"
Property_file = ""

Result_Dir = Wspace+"Res/"

trans_slope_all = 90       # [%] Max cross slope outside hairpin
trans_slope_hairpin = 55   # [%] Max cross slope at hairpin
min_slope = 2              # [%] Min slope of the road
max_slope = 12             # [%] Max slope of the road
penalty_xy = 150           # [m/180°] Turn around penalty
penalty_z = 80             # [m/2*max(min_slope,max_slope)] "Wave" penalty
D_neighborhood = 42        # [m] Distance of the viciny around pixel
max_diff_z = 3             # [m] Max difference of elevation between terrain and
                           #     theoretical elevation of the road 
angle_hairpin = 110        # [°] Min angle for a turn to be considered as hairpin
Lmax_ab_sl = 40            # [m] Max length of road with cross slope > trans_slope_all
Radius = 8                 # [m] Radius for trucks

################################################################################
### Script execution
################################################################################
if 'Wspace' not in locals() or 'Wspace' not in globals() :
    Wspace = Result_Dir


road_finder_exec_force_wp(Dtm_file,Obs_Dir,Waypoints_file,Property_file,Result_Dir,
                          trans_slope_all,trans_slope_hairpin,min_slope,max_slope,
                          penalty_xy,penalty_z,D_neighborhood,max_diff_z,
                          angle_hairpin,Lmax_ab_sl,Wspace,Radius)


