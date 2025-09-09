import numpy as np
from constants import * 


dirf = 'Data/Gaia Challenge/' #directory for data
diro = 'Output/Gaia Challenge/' #directory for Output

vdata = '1krltr_vels_plumcoreom.txt' #1k tracers (LOS + PMs)
#vdata = '100rltr_vels_plumcoreom.txt' #100 tracers (LOS + PMs)
#vdata = '10krltr_vels_plumcoreom.txt' #10k tracers (LOS + PMs)


rLOS, vLOS, vLOS_err, vPMt, vPMt_err, vPMR, vPMR_err = np.loadtxt(str(dirf + vdata)) #LOS + proper-motion position + velocity data (1k tracers)
rPM = rLOS

R, surfden, surfdenerr = np.loadtxt(str(dirf + 'photplumcore.txt')) #photometric tracer profile

propermotion = True #fitting proper motions?

individual = False #individual star positions (R) or binned profile for photometry (R, surfden, surfden_err)?

Rhalf = 0.2760552110422084 #half-light radius (doesn't need to be exact, but rather used as a reference scale)

esc_check = False #include escape velocity condition? Potentially useful for low tracer numbers or if individual velocities differ substantially 
#from velocity dispersion. Assumes stars should be members, if this is not a good assumption, then remove problematic stars or use a mixture model where these are moved to foreground stars

#Radial grid range for Jeans calculation:
rmin = np.min([Rhalf / 100, np.min(R) / 10])
rmax = np.max([Rhalf * 100, np.max(R) * 2])

rcn = rmax / 2.0 #ensure non-negative moments up to this radius


Mstar = 0.0 #massless tracer / stellar profile, i.e. DM-only
barrad_min = rmin
barrad_max = rmax
bar_pnts = 300

###########################################################
#Priors


pfits = np.array([13, 0.25,2,5,0.1]) #reference values of alpha-beta-gamma profile obtained from pre-fitting (rho0, r0, alp, bet, gam) this can be #done e.g. with binulator, directly fitting or manually (e.g. looking at a Plummer profile matching central density and ~ Rhalf for the scale radius)
tracertol = 0.5 #tolerance for flat priors around best-fit / reference tracer profile values
#Alternatively, one can use broad (e.g. logarithmic) priors, which can be done by modifying the gravsphere2 file

#Symmetrized velocity anisotropy priors:
betr0min = -2
betr0max = 0.0
betnmin = 1.0
betnmax = 3.0
bet0min = -0.01
bet0max = 0.01
betinfmin = -0.1
betinfmax = 1.0

#CoreNFWtides priors:
logM200low = 7.5
logM200high = 11.5
clow = 1.0
chigh = 100.0
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
