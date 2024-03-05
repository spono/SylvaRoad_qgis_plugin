# -*- coding: utf8 -*-
"""
Software: SylvaRoaD
File: SylvaRoaD_1_cython.pyx
Copyright (C) Sylvain DUPIRE - SylvaLab 2021
Authors: Sylvain DUPIRE
Contact: sylvain.dupire@sylvalab.fr
Version: 2.2
Date: 2021/08/09
License :  GNU-GPL V3
"""
import numpy as np
cimport numpy as np
import cython
cimport cython

##############################################################################################################################################
### Fonctions generales
##############################################################################################################################################
cdef extern from "math.h":
    double acos(double)
    double atan(double)
    double asin(double)
    double sqrt(double)
    double cos(double)
    double fabs(double)
    double log(double)
    double cos(double)
    double tan(double)
    double sin(double) 
    double sinh(double)
    double cosh(double)
    int floor(double)
    int ceil(double)    
    double atan2(double a, double b)
    double degrees(double)
    bint isnan(double x)


cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)

cdef extern from "Python.h":
    Py_ssize_t PyList_Size(object) except -1   #Equivalent de len(list)
    int PyList_Append(object, object) except -1 #Equivalent de list.append()
    object PyList_GetSlice(object, Py_ssize_t low, Py_ssize_t high) #Equivalent de list[low:high]
    int PyList_Sort(object) except -1 #Equivalent de list.sort()

ctypedef np.int_t dtype_t
ctypedef np.float_t dtypef_t
ctypedef np.int8_t dtype8_t
ctypedef np.uint8_t dtypeu8_t
ctypedef np.uint16_t dtypeu16_t
ctypedef np.int16_t dtype16_t
ctypedef np.int32_t dtype32_t
ctypedef np.int64_t dtype64_t
ctypedef np.float32_t dtypef32_t

# Retourne le maximum d'un array
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int max_array(np.ndarray[dtype_t, ndim=1] a):
    cdef int max_a = a[0]
    cdef unsigned int item = 1
    for item from 1 <= item < a.shape[0]:
        if a[item] > max_a:max_a = a[item]
    return max_a

# Retourne le maximum d'un array de float
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double max_array_f(np.ndarray[dtypef_t, ndim=1] a):
    cdef double max_a = a[0]
    cdef unsigned int item = 1
    for item from 1 <= item < a.shape[0]:
        if a[item] > max_a:max_a = a[item]
    return max_a

# Retourne la somme d'un array
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int sum_array(np.ndarray[dtype8_t, ndim=1] a):
    cdef int summ = a[0]
    cdef unsigned int item = 1
    for item from 1 <= item < a.shape[0]:
        summ += item
    return summ

# Definie la fonction arcsinus hyperbolique
cdef inline double asinh(double x):return log(x+sqrt(1+x*x))
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline double double_min(double a, double b): return a if a <= b else b
cdef inline double double_max(double a, double b): return a if a >= b else b
cdef inline double square(double x):return x*x
cdef inline int cint(double x):return int(x)
#cdef inline int isnan(value):return 1 if value != value else 0


cdef double pi=3.141592653589793

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef float modulo(float a,float b):
    cdef float val,ent
    if a<b and a>=0:
        val=a
    elif a<b and a<0:
        ent = cint(a/b)
        val = b+(a-ent*b)
    else:
        ent = cint(a/b)
        val = a-ent*b
    return val


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef float calculate_azimut(float x1,float y1,float x2,float y2):   
    cdef float DX = x2-x1
    cdef float DY = y2-y1
    cdef float Deuc = sqrt(DX*DX+DY*DY)
    cdef int Fact=-1
    cdef float Angle
    if x2>x1:Fact=1
    Angle = acos(DY/Deuc)*180/pi
    Angle *=Fact
    return modulo(Angle,360)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double conv_az_to_polar(double az):
    cdef double val
    val = (360-(az-90))
    return modulo(val,360)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double diff_az(double az_to,double az_from):   
    if az_to>az_from:
        return min((360-(az_to-az_from),(az_to-az_from)))
    else:
        return min((360-(az_from-az_to),(az_from-az_to)))



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double check_focal_nb(np.ndarray[dtypef32_t,ndim=2] raster, 
                          double rayon,double Csize,int x1,int y1,
                          int nline, int ncol, double trans_slope_hairpin):       
    cdef int cote = int(rayon/Csize)
    cdef int nbsup=0     
    cdef int y,x,nb=0
    # Grille sans les bordures   
    for y in range(max(0,y1-cote),min(nline,y1+cote+1)):
        for x in range(max(0,x1-cote),min(ncol,x1+cote+1)):
            if sqrt((x1-x)**2+(y1-y)**2)*Csize>rayon:
                continue  
            if raster[y,x]!=-9999:
                nb+=1
            if raster[y,x]>trans_slope_hairpin:
                nbsup+=1
    return 1.*nbsup/nb

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[dtypeu8_t,ndim=2] calc_local_slope(np.ndarray[dtypef32_t,ndim=2] raster,
                                                   double rayon,double Csize,
                                                   double trans_slope_hairpin):
                                        
    cdef int nline=raster.shape[0],ncol = raster.shape[1]
    cdef int y,x
    cdef np.ndarray[dtypeu8_t,ndim=2] local_slope = np.ones((nline,ncol),dtype=np.uint8)*255
    for y in range(nline):
        for x in range(ncol):
            if raster[y,x]!=-9999:
                local_slope[y,x]=int(100*check_focal_nb(raster,rayon,Csize,x,y,
                                                nline, ncol, trans_slope_hairpin)+0.5)
    return local_slope
            
    
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int check_ptbef(np.ndarray[dtype32_t,ndim=2] pt_neig,
                      np.ndarray[dtype32_t,ndim=2] pt_current):
                                        
    cdef int test=0 
    if pt_neig[1,0]== pt_current[0,0]:
        test+=1    
    if test>0 and pt_neig[1,1]== pt_current[0,1]:
        test+=1
    if test>1 and pt_neig[2,0]== pt_current[1,0]:
        test+=1
    if test>2 and pt_neig[2,1]== pt_current[1,1]:
        test+=1
    if test==4:
        test=1
    return test

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int get_intersect(double a1y,double  a1x,double  a2y,double  a2x,
                   double b1y,double  b1x,double  b2y,double  b2x):
    cdef double  l1y,l1x,l1z,l2y,l2x,l2z,y,x,z,xi,yi
    cdef int inter = 1
    l1y = a1x-a2x
    l1x = a2y-a1y
    l1z = a1y*a2x-a1x*a2y
    
    l2y = b1x-b2x
    l2x = b2y-b1y
    l2z = b1y*b2x-b1x*b2y
    
    y = l1x*l2z-l1z*l2x
    x = l1z*l2y-l1y*l2z
    z = l1y*l2x-l1x*l2y
   
    if z == 0:                          # lines are parallel
        inter = 0
    if z!=0:
        xi,yi = x/z,y/z
        if xi < double_max(double_min(a1x,a2x),double_min(b1x,b2x)) :
            inter = 0
        elif xi > double_min(double_max(a1x,a2x),double_max(b1x,b2x)):
            inter = 0
        if yi < double_max(double_min(a1y,a2y),double_min(b1y,b2y)) :
            inter = 0
        elif yi > double_min(double_max(a1y,a2y),double_max(b1y,b2y)):
            inter = 0
    return inter

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int check_intersect(np.ndarray[dtype32_t,ndim=2] Line_before, np.ndarray[dtype_t,ndim=2] Line_after,int nbptbef):
    cdef int a1x,a1y,a2x,a2y,b1x,b1y,b2x,b2y,i
    cdef int inter=0
    b1y,b1x = Line_after[0,0],Line_after[0,1]
    b2y,b2x = Line_after[1,0],Line_after[1,1]
    for i in range(nbptbef-1):
        a1y,a2y = Line_before[i,0],Line_before[i+1,0]
        a1x,a2x = Line_before[i,1],Line_before[i+1,1]
        inter = get_intersect(a1y, a1x, a2y, a2x,b1y, b1x, b2y, b2x)
        if inter:
            break
    return inter


cpdef double Distplan(double y, double x, double yE, double xE):
    return sqrt((y-yE)*(y-yE)+(x-xE)*(x-xE))

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef connect2(int yc,int xc,int y, int x):
    cdef int d0 = y-yc
    cdef int d1 = x-xc
    cdef int sign = 1      
    cdef np.ndarray[dtype32_t,ndim=1] ys,xs
        
    if abs(d0) > abs(d1): 
        if d0<0:
            sign = -1
        ys = np.arange(yc, y + sign, sign, dtype=np.int32)  
        if d1==0:
            xs = np.ones_like(ys,dtype=np.int32)*xc
        else:
            xs = np.int32(np.arange(xc * abs(d0) + floor(abs(d0)/2),
                           xc * abs(d0) + floor(abs(d0)/2) + (abs(d0)+1) * d1,
                           d1, dtype=np.int32) / abs(d0))
    else:
        if d1<0:
            sign = -1
        xs = np.arange(xc, x + sign, sign, dtype=np.int32)
        if d0==0:            
            ys = np.ones_like(xs,dtype=np.int32)*yc
        else:
            ys = np.int32(np.arange(yc * abs(d1) + floor(abs(d1)/2),
                           yc * abs(d1) + floor(abs(d1)/2) + (abs(d1)+1) * d0, 
                           d0, dtype=np.int32) / abs(d1))
    return ys,xs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef check_profile(int yc,int xc,int y, int x, double slope_perc,
                   np.ndarray[dtypef_t,ndim=2] dtm,double Csize,
                   double max_diff_z,np.ndarray[dtype8_t,ndim=2] Obs,
                   np.ndarray[dtype8_t,ndim=2] Obs2, double Ls, double Lmax_ab_sl):
    cdef np.ndarray[dtype32_t,ndim=1] ys,xs
    cdef int test=1,nbpix = 0,i,sumobs2=0
    cdef double newLsl=Ls,z,zline,Dhor,zo,diffz = 0
     
    ys,xs = connect2(yc,xc,y,x) 
    nbpix = ys.shape[0]
    zo = dtm[yc,xc]
    for i in range(nbpix):
        if Obs[ys[i],xs[i]]>0:
            test=0
            break
        Dhor=sqrt((xs[i]-xc)*(xs[i]-xc)+(ys[i]-yc)*(ys[i]-yc))*Csize
        if i>0:
            sumobs2+=Obs2[ys[i],xs[i]]
        z=dtm[ys[i],xs[i]]
        zline = slope_perc/100.*Dhor+zo
        diffz = double_max(diffz,abs(zline-z))
        if diffz>max_diff_z:
            test=0
            break
    if test:
        newLsl += double_min(sumobs2*Csize,Dhor)
        if newLsl>Lmax_ab_sl:                      
            test=0        
    return test,newLsl
    

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double diffz_prop_L(double max_diff_z,double D_neighborhood,double L):
    return max_diff_z*L/D_neighborhood

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef build_Tab_neibs(np.ndarray[dtype8_t,ndim=2] newObs,
                      np.ndarray[dtypef_t,ndim=2] dtm,
                      np.ndarray[dtypef32_t,ndim=1] azimuts,
                      np.ndarray[dtypef32_t,ndim=1] dists_index,
                      np.ndarray[dtype32_t,ndim=2] coords,
                      double min_slope,
                      double max_slope,int nbpix):
    
    cdef int x,y,x1,y1,ind=0,nbok=0,i,sign
    cdef int nrows=dtm.shape[0],ncols=dtm.shape[1]
    cdef int nbneig = azimuts.shape[0]
    cdef float D,sl,deltaH,z,z1,abssl
    
    cdef np.ndarray[dtype32_t,ndim=2] IdPix = np.ones_like(dtm,dtype=np.int32)*-1     
    cdef np.ndarray[dtypeu16_t,ndim=2] IdVois = np.zeros((nbpix,nbneig),dtype=np.uint16)
    cdef np.ndarray[dtype16_t,ndim=2] Slope = np.ones((nbpix,nbneig),dtype=np.int16)*-9999   
    cdef np.ndarray[dtype32_t,ndim=2] Id = np.ones((nbpix,nbneig),dtype=np.int32)*-9999
    cdef np.ndarray[dtypeu16_t,ndim=2] Tab_corresp = np.zeros((nbpix,3),dtype=np.uint16)
    
    for y in range(nrows):
        for x in range(ncols):
            if newObs[y,x]==0:
                IdPix[y,x]=ind
                Tab_corresp[ind,0]=y
                Tab_corresp[ind,1]=x          
                ind+=1
                
    for ind in range(nbpix):  
        y = Tab_corresp[ind,0]
        x = Tab_corresp[ind,1]
        nbok=0
        z=dtm[y,x]
        for i in range(nbneig):
            y1=y+coords[i,0]
            x1=x+coords[i,1]
            
            if y1<0 or y1>=nrows or x1<0 or x1>=ncols:
                continue                    
            if newObs[y1,x1]:
                continue
            if y1==y and x1==x:
                continue
            z1=dtm[y1,x1]
            deltaH = z1-z 
            D=dists_index[i]
            sl=deltaH/D*100
            abssl = abs(sl)                  
            if abssl>=min_slope and abssl<=max_slope:
                IdVois[ind,nbok] = i
                Id[ind,nbok]=IdPix[y1,x1]
                sign=1
                if sl<0:
                    sign=-1
                Slope[ind,nbok]=int(abssl*100+0.5)*sign
                nbok+=1
        Tab_corresp[ind,2]=nbok
            
    return IdVois, Id, Tab_corresp ,IdPix ,Slope




@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calc_init(int idcurrent,
                np.ndarray[dtype32_t,ndim=2] Id,
                np.ndarray[dtypeu16_t,ndim=2] IdVois,
                np.ndarray[dtype16_t,ndim=2] Slope,
                np.ndarray[dtypef32_t,ndim=2] Best,
                np.ndarray[dtypeu16_t,ndim=2] Tab_corresp,
                np.ndarray[dtypef32_t,ndim=1] Az,
                np.ndarray[dtypef32_t,ndim=1] Dist,
                np.ndarray[dtype8_t,ndim=2] newObs,
                np.ndarray[dtype8_t,ndim=2] Obs2,
                np.ndarray[dtypef32_t,ndim=2] Dist_to_End,
                np.ndarray[dtypef_t,ndim=2] dtm,            
                double Csize,double max_diff_z,
                double D_neighborhood,double Lmax_ab_sl,
                int take_dtoend,int yE,int xE,double mindist_to_end):
    cdef int xc = Tab_corresp[idcurrent,1]
    cdef int yc =Tab_corresp[idcurrent,0]  
    cdef int nbvois = Tab_corresp[idcurrent,2]  
    cdef np.ndarray[dtype32_t,ndim=1] add_to_frontier = np.zeros((nbvois,),dtype=np.int32)
    cdef int nbok,neib,idvois,y,x,test_prof
    cdef float D,Azimut,slope_perc,newLsl,D_to_cp
    
    nbok=0      
    for neib in range(nbvois):
        idvois = Id[idcurrent,neib]
        D = Dist[IdVois[idcurrent,neib]]        
        y = Tab_corresp[idvois,0]
        x = Tab_corresp[idvois,1]
        if newObs[y,x]:#The pixel is an Obstacle
            continue
        D = Dist[IdVois[idcurrent,neib]]        
        Azimut = Az[IdVois[idcurrent,neib]]
        slope_perc = Slope[idcurrent,neib]/100.
        
        test_prof,newLsl = check_profile(yc,xc,y,x,slope_perc,dtm,Csize,
                                         diffz_prop_L(max_diff_z,D_neighborhood,D),
                                         newObs,Obs2,0,Lmax_ab_sl)
        if not test_prof:                        
            continue 
       
        if take_dtoend:
            D_to_cp = Dist_to_End[y,x]
        else:
            D_to_cp = Distplan(y,x,yE,xE)*Csize                                      
                                
        add_to_frontier[nbok]=idvois
        nbok+=1
        Best[idvois,0] = idvois
        Best[idvois,1] = D+newLsl
        Best[idvois,2] = D    
        Best[idvois,3] = slope_perc
        Best[idvois,4] = Azimut
        Best[idvois,5] = idcurrent
        Best[idvois,6] = 10*D_neighborhood        
        Best[idvois,7] = newLsl
        Best[idvois,8] = 1   
        Best[idvois,9] = D_to_cp              
        
        mindist_to_end = min(mindist_to_end,D_to_cp) 

    return Best,add_to_frontier[0:nbok],mindist_to_end

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef basic_calc(int idcurrent,
                 np.ndarray[dtype32_t,ndim=2] Id,
                 np.ndarray[dtypeu16_t,ndim=2] IdVois,
                 np.ndarray[dtype16_t,ndim=2] Slope,
                 np.ndarray[dtypef32_t,ndim=2] Best,
                 np.ndarray[dtypeu16_t,ndim=2] Tab_corresp,
                 np.ndarray[dtypef32_t,ndim=1] Az,
                 np.ndarray[dtypef32_t,ndim=1] Dist,
                 np.ndarray[dtype8_t,ndim=2] newObs,
                 np.ndarray[dtype8_t,ndim=2] Obs2,
                 np.ndarray[dtypef32_t,ndim=2] Dist_to_End,
                 np.ndarray[dtypef_t,ndim=2] dtm,
                 double LocSlope, double Csize,
                 double max_diff_z,double D_neighborhood,
                 double Lmax_ab_sl,int take_dtoend,
                 int yE,int xE,double mindist_to_end,
                 double prop_sl_max,double angle_hairpin,
                 double Radius,double penalty_xy,
                 double penalty_z,double max_slope_change,
                 double max_hairpin_angle,double Dmin,
                 int max_nbptbef,float modhair=1.5):
    
    cdef int xc = Tab_corresp[idcurrent,1]
    cdef int yc =Tab_corresp[idcurrent,0]  
    cdef int nbptbef = int(Best[idcurrent,8])
    cdef int nbvois = Tab_corresp[idcurrent,2]  
    cdef np.ndarray[dtype32_t,ndim=1] add_to_frontier = np.zeros((nbvois,),dtype=np.int32)    
    cdef int nbok,neib,idvois,y,x,test_prof,hairpin,i,idfrom2
    cdef float D,Azimut,slope_perc,newLsl,D_to_cp,difangle2,difangle,Dcurrent
    cdef float penalty_dir,difslope,penalty_slope,new_cost,az2,az1,ycen,xcen
    cdef int a1x,a1y,a2x,a2y,inter,idfrom  
    cdef float pen_hp=100*(LocSlope/prop_sl_max)**2  
    nbok=0    
     
    for neib in range(nbvois):
        idvois = Id[idcurrent,neib]
        D = Dist[IdVois[idcurrent,neib]]
        if Best[idvois,1]<Best[idcurrent,1]+D:
            continue 
        y = Tab_corresp[idvois,0]
        x = Tab_corresp[idvois,1]
        if newObs[y,x]:#The pixel is an Obstacle
            continue
        if idvois==Best[idcurrent,5]:
            continue  
        
        difangle2,hairpin =0,0
        Azimut = Az[IdVois[idcurrent,neib]]
        slope_perc = Slope[idcurrent,neib]/100.
        
        difangle = diff_az(Azimut,Best[idcurrent,4])               
        #turn around
        if difangle>max_hairpin_angle:
            continue
        #hairpin in 1 turn
        if difangle>angle_hairpin :
            if LocSlope>prop_sl_max: 
                continue
            hairpin=1
        #hairpin in 2 turns    
        if nbptbef>1 and not hairpin: 
            idfrom = int(Best[idcurrent,5])   
            Dcurrent = Best[idcurrent,2]-Best[idfrom,2] 
            difangle2 = diff_az(Azimut,Best[idfrom,4])
            if Dcurrent<=2*Radius and difangle2>angle_hairpin :
                if LocSlope>prop_sl_max:
                    continue           
                ######
                ### modif ici
                ######   
                ycen=0.5*(yc+Tab_corresp[idfrom,0])
                xcen=0.5*(xc+Tab_corresp[idfrom,1])
                idfrom2 = int(Best[idfrom,5])
                a2y = Tab_corresp[idfrom2,0]
                a2x = Tab_corresp[idfrom2,1]
                az1 = calculate_azimut(a2x,a2y,xcen,ycen)
                az2 = calculate_azimut(xcen,ycen,x,y)
                difangle2 = diff_az(az1,az2)
                if difangle2>max_hairpin_angle:
                    continue
                hairpin=1                    
        #Check if there is hairpin in the last 3*Radius meters
        if hairpin and Best[idcurrent,6]<= 2*modhair*Radius: 
            continue              
        #Check if there is intersection with previous road  
        if nbptbef>1:
            inter=0
            i=1
            idfrom = int(Best[idcurrent,5])   
            while i<nbptbef:
                a1y=Tab_corresp[idfrom,0]
                a1x=Tab_corresp[idfrom,1]   
                idfrom = int(Best[idfrom,5]) 
                a2y=Tab_corresp[idfrom,0]
                a2x=Tab_corresp[idfrom,1]    
                if get_intersect(a1y, a1x, a2y, a2x,yc, xc, y, x):
                    inter=1
                    break
                i+=1
            if inter:
                continue            
            
        penalty_dir = penalty_xy*(max(difangle,difangle2)/angle_hairpin)**2                    
        difslope = abs(Best[idcurrent,3] - slope_perc)
        penalty_slope = penalty_z*(difslope/max_slope_change)**2

        test_prof,newLsl =check_profile(yc,xc,y,x,slope_perc,dtm,Csize,
                                        diffz_prop_L(max_diff_z,D_neighborhood,D),
                                        newObs,Obs2,Best[idcurrent,7],Lmax_ab_sl)
        if not test_prof:                        
            continue 
        
        new_cost = Best[idcurrent,1]+D+penalty_dir+penalty_slope+newLsl-Best[idcurrent,7]     
        if hairpin:
            new_cost+=pen_hp
        
        if take_dtoend:
            D_to_cp = Dist_to_End[y,x]
        else:
            D_to_cp = Distplan(y,x,yE,xE)*Csize                                        
                                
        if Best[idvois,1] > new_cost : 
            add_to_frontier[nbok]=idvois
            nbok+=1
            # if idcurrent==Best[idvois,6]:
            #     print(idvois,idcurrent,Best[idcurrent,6],Best[idvois,2],Best[idcurrent,2]+Dist )            
            Best[idvois,0] = idvois
            Best[idvois,1] = new_cost 
            Best[idvois,2] = Best[idcurrent,2]+D    
            Best[idvois,3] = slope_perc
            Best[idvois,4] = Azimut
            Best[idvois,5] = idcurrent             
            Best[idvois,7] = newLsl
            Best[idvois,8] = Best[idcurrent,8]+1   
            Best[idvois,9] = D_to_cp 
            Best[idvois,10] = hairpin
            #### calculate nearest distance to hairpin 
            Best[idvois,6] = 10*D_neighborhood
            if hairpin:
                Best[idvois,6] = min(Best[idvois,6],D)            
            i=1
            idfrom = idcurrent             
            while i<nbptbef-1 and Best[idvois,6]>=2*modhair*Radius:
                idfrom2 = int(Best[idfrom,5]) 
                if Best[idfrom,10]:
                    a1y=Tab_corresp[idfrom2,0]
                    a1x=Tab_corresp[idfrom2,1]  
                    Best[idvois,6] = min(Best[idvois,6],Distplan(y,x,a1y,a1x)*Csize   )                     
                idfrom=idfrom2
                i+=1 
            mindist_to_end = min(mindist_to_end,D_to_cp) 
            
    return Best,add_to_frontier[0:nbok],mindist_to_end


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)                   
cpdef get_pix_bufgoal_and_update(np.ndarray[dtypef32_t,ndim=2] Best,
                                 np.ndarray[dtypeu16_t,ndim=2] Tab_corresp,
                                 double bufgoal,int start,double Csize,
                                 int yE, int xE):    
    
    cdef int nbmax = (2*int(bufgoal/Csize+0.5)+1)*(2*int(bufgoal/Csize+0.5)+1)
    cdef int nbval = Best.shape[0]
    cdef np.ndarray[dtype32_t,ndim=1] add_to_frontier = np.zeros((nbmax,),dtype=np.int32)  
    cdef np.ndarray[dtypeu8_t,ndim=1] keep = np.zeros((nbval,),dtype=np.uint8) 
    
    cdef int i,j=0,y,x
    cdef int current 
    
    for i in range(nbval):
        if Best[i,0]<0:
            continue
        if Best[i,9]<=bufgoal:
            add_to_frontier[j]=i            
            j+=1
            current = i
            #mark pixel from as 1
            while current != start:
                if keep[current]==1:
                    break
                keep[current]=1
                current = int(Best[current,5])
        y = Tab_corresp[i,0]
        x = Tab_corresp[i,1]
        Best[i,9] = Distplan(y,x,yE,xE)*Csize  
    
    return Best,add_to_frontier[0:j],keep


                
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)                   
cpdef calcul_distance_de_cout(int yE,int xE,
                            np.ndarray[dtype8_t,ndim=2] zone_rast,
                            double Csize,double Max_distance=100000):    
    
    cdef np.ndarray[dtype8_t,ndim=2] coords = np.array([[-1, -1],[ 0, -1],[ 1, -1], 
                                                 [-1,  0],[ 1,  0],
                                                 [-1,  1],[ 0,  1],[ 1,  1]],dtype=np.int8)    
    
    cdef int nbneig = coords.shape[0]
    cdef int  nline = zone_rast.shape[0]
    cdef int ncol = zone_rast.shape[1]
    cdef np.ndarray[dtypef32_t,ndim=1] dists_index = np.zeros((nbneig,),dtype=np.float32)
          
    # Creation des rasters de sorties
    cdef np.ndarray[dtypef32_t,ndim=2] Out_distance = np.ones_like(zone_rast,dtype=np.float32)*Max_distance
    cdef np.ndarray[dtype32_t,ndim=2] Inds = np.ones((nline*ncol,2),dtype=np.int32)*-9999   
    cdef np.ndarray[dtype32_t,ndim=2] Indsbis     
    # Initialisation du raster
    cdef int i,j,y,x
    cdef float Dist,dist_ac
    cdef int x1=xE,y1=yE
    cdef int nbinds,idpix
    for j in range(nbneig):
        dists_index[j]=sqrt(coords[j,0]*coords[j,0]+coords[j,1]*coords[j,1])*Csize    
    
    Out_distance[y1,x1] = 0
    i=0
    for j in range(nbneig):
        y = coords[j,0]+y1
        x = coords[j,1]+x1
        if y<0 or y>=nline or x<0 or x>=ncol:continue
        if zone_rast[y,x]==1:  
            Dist = dists_index[j]
            if Out_distance[y,x]>Dist:
                Out_distance[y,x] = Dist                              
                Inds[i,0],Inds[i,1] = y,x
                i+=1
            
    # Traitement complet   
    while i>0:
        Indsbis = np.copy(Inds[0:i])       
        nbinds = i
        i=0
        
        for idpix in range(nbinds):            
            y1,x1=Indsbis[idpix,0],Indsbis[idpix,1]
            dist_ac = Out_distance[y1,x1]
            for j in range(nbneig):
                y = coords[j,0]+y1
                x = coords[j,1]+x1
                if y<0 or y>=nline or x<0 or x>=ncol:continue
                if zone_rast[y,x]==1: 
                    dist_ac = Out_distance[y1,x1] +dists_index[j]                   
                    if Out_distance[y,x]>dist_ac:                       
                        Out_distance[y,x] =dist_ac
                        Inds[i,0],Inds[i,1] = y,x
                        i+=1
        
    for y in range(0,nline,1):
        for x in range(0,ncol,1):
            if Out_distance[y,x]==Max_distance:
                Out_distance[y,x]=-9999
    return Out_distance


