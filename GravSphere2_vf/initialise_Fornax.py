import numpy as np
from constants import * 

dirf = 'Data/Fornax Simulation/' #directory for data
diro = 'Output/Fornax Simulation/' #directory for Output

#vdata = 'forall1k.txt' #1k tracers (LOS + PMs)

vdata = 'forall100.txt' #100 tracers (LOS + PMs)

#vdata = 'fornaxlos50.txt' #50 LOS tracers

#vdata = 'fornaxlos10.txt' #10 LOS tracers

#for LOS velocities-only use 'fornaxlos.txt' (all tracers), 'fornaxlos1k.txt' (1k tracers),
#and 'fornaxlos100.txt' (100 tracers) to match the same data / definition of LOS as Tchiorniy & Genina 2025, in this case,
#you can do rLOS, vLOS, vLOS_err = np.loadtxt(dirf + vdata) and propermotion = False (no PMs). We've also included the 10, 25, 50 tracer files.

rLOS, vLOS, vLOS_err, vPMt, vPMt_err, vPMR, vPMR_err = np.loadtxt(str(dirf + vdata)) #LOS + proper-motion position + velocity data (1k tracers)
rPM = rLOS

#rLOS, vLOS, vLOS_err = np.loadtxt(str(dirf + vdata)) #LOS + proper-motion position + velocity data (1k tracers)
#rPM = rLOS

R = np.loadtxt(str(dirf + 'fornax_phot_10k.txt')) #photometric tracer positions for individual fitting (10k tracers)
#R = rLOS #use kinematic tracers as photometric ones

propermotion = True #fitting proper motions?

individual = True #individual star positions (R) or binned profile for photometry (R, surfden, surfden_err)?

Rhalf = np.median(R) #half-light radius

esc_check = False #include escape velocity condition? Potentially useful for low tracer numbers or if individual velocities differ substantially 
#from velocity dispersion. Assumes stars should be members, if this is not a good assumption, then remove problematic stars or use a mixture model where these are moved to foreground stars

#Radial grid range for Jeans calculation:
rmin = np.min([Rhalf / 100, np.min(R) / 10])
rmax = np.max([Rhalf * 100, np.max(R) * 2])

rcn = rmax / 2.0

#Galaxy properties. Assume here that the baryonic mass
#has the same radial profile as the tracer stars. If this
#is not the case, you should set Mstar_rad and Mstar_prof 
#here. The variables barrad_min, barrad_max and bar_pnts 
#set the radial range and sampling of the baryonic mass model.

Mstar = 4.3e7
Mstar_err = Mstar * 0.25
Mstar_min = Mstar - Mstar_err
Mstar_max = Mstar + Mstar_err
baryonmass_follows_tracer = 'yes'
barrad_min = rmin
barrad_max = rmax
bar_pnts = 300

###########################################################
#Priors

#Symmetrized velocity anisotropy priors:
betr0min = -2
betr0max = 0.0
betnmin = 1.0
betnmax = 3.0
bet0min = -1.0
bet0max = 1.0
betinfmin = -1.0
betinfmax = 1.0

#CoreNFWtides priors:
logM200low = 7.5
logM200high = 11.5
clow = 1.0
chigh = 50.0
rclow = 1e-2
rchigh = 10.0
logrclow = np.log10(rclow)
logrchigh = np.log10(rchigh)
nlow = 0.0
nhigh = 1.0
rtlow = 1.0
rthigh = 20.0
logrtlow = np.log10(rtlow)
logrthigh = np.log10(rthigh)
dellow = 3.01
delhigh = 5.0
