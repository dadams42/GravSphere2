###########################################################
#GravSphere 2
###########################################################

#Choose the specific object with initialisation file defining priors, Rhalf, velocities and photometry data 

from initialise_Fornax import *
#from initialise_PlumCoreOm import *
#from initialise_PlumCuspOm import *

#imports functions for velocity PDF modeling from (Sanders & Evans 2020), link: https://arxiv.org/abs/2009.07858
#link to J. Sanders's GitHub repository: https://github.com/jls713/gh_alternative

from line_profiles import *

from scipy.special import gamma
import numpy as np
from scipy.integrate import simpson as integrator
#from scipy.integrate import simps as integrator #use for older versions of scipy
from constants import *
from gs2_functions import *
import scipy

#Suppress warning output:
import warnings
warnings.simplefilter("ignore")


##            

#sets priors including prior transform implementation for dynesty
if individual == False: #using photometric profile fit (otherwise fits individual stellar positions)

        #pfits are pre-fitted parameters (with e.g. binulator) for setting the priors on the alpha-beta-gamma model for photometry

        #should specify #Rphot, surfden, surfdenerr in initialise file (radius and surface density + errors)
    
        nupars_min = np.zeros(len(pfits))
        nupars_max = np.zeros(len(pfits))

        for i in range(len(pfits)):
            if (pfits[i] > 0):
                nupars_min[i] = pfits[i]*(1.0-tracertol)
                nupars_max[i] = pfits[i]*(1.0+tracertol)
            else:
                nupars_min[i] = pfits[i]*(1.0+tracertol)
                nupars_max[i] = pfits[i]*(1.0-tracertol)

        nu_components = len(pfits) # number of free parameter for photometry

        #To ensure positivity and convergence, we need alp > 0, bet > 3, gamm < 3 
        nupars_min[2:4] = np.maximum(nupars_min[2:4], np.array([0.0, 3.0]))
        nupars_max[4] = np.min(np.array([nupars_max[4], 3.0]))
 



else: #photometric profile fit (fits individual stellar positions, most accurate if available)
        
        #general priors on #r0,alp,bet,gam (rh0 is not needed from probability normalization)
        #only R is needed (i.e. positions)
       #Plummer corresponds to alp, bet, gam = 2, 5, 0.0
       #To ensure positivity and convergence, we need alp > 0, bet > 3, gamm < 3
        nupars_min = np.array([np.log10(Rhalf) - 1.0, 0.1, 3.1, 0.0])
        nupars_max = np.array([np.log10(Rhalf) + 1.0, 4.0, 7.0, 2.9])

        nu_components = 4

n_betpars = 4 # number of parameters for generalised OM anisotropy

ndims = 2 * n_betpars + nu_components + 6

if Mstar > 0.0:

    ndims += 1

#anisotropy priors
minarr = np.zeros(ndims)
maxarr = np.zeros(ndims)
minarr[0] = bet0min
maxarr[0] = bet0max
minarr[1] = betinfmin
maxarr[1] = betinfmax
minarr[2] = betr0min
maxarr[2] = betr0max
minarr[3] = betnmin
maxarr[3] = betnmax
minarr[4] = bet0min
maxarr[4] = bet0max
minarr[5] = betinfmin
maxarr[5] = betinfmax
minarr[6] = betr0min
maxarr[6] = betr0max
minarr[7] = betnmin
maxarr[7] = betnmax
minarr[2*n_betpars:2*n_betpars+nu_components] = nupars_min
maxarr[2*n_betpars:2*n_betpars+nu_components] = nupars_max
minarr[2*n_betpars+nu_components] = logM200low
maxarr[2*n_betpars+nu_components] = logM200high
minarr[2*n_betpars+nu_components+1] = clow
maxarr[2*n_betpars+nu_components+1] = chigh
minarr[2*n_betpars+nu_components+2] = logrclow
maxarr[2*n_betpars+nu_components+2] = logrchigh
minarr[2*n_betpars+nu_components+3] = nlow
maxarr[2*n_betpars+nu_components+3] = nhigh
minarr[2*n_betpars+nu_components+4] = logrtlow
maxarr[2*n_betpars+nu_components+4] = logrthigh
minarr[2*n_betpars+nu_components+5] = dellow
maxarr[2*n_betpars+nu_components+5] = delhigh

#log-variables for sampling
logthis = [2, 6, 2*n_betpars+nu_components, 2*n_betpars+nu_components + 2, 2*n_betpars+nu_components+4] 


if individual == True:

    logthis.append(2*n_betpars)

#add mass of stellar component if applicable
if Mstar > 0.0:

    minarr[2*n_betpars+nu_components+5 + 1] = Mstar_min
    maxarr[2*n_betpars+nu_components+5 + 1] = Mstar_max

#prior transform for dynesty
def ptform(u):
    
    x = np.array(u)
    
    return minarr + (maxarr - minarr) * x

if (barrad_min == 0):
        barrad_min = 1.0e-3
    
#sets-up radial range for tracer profile numerics    
nu_rad = np.logspace(np.log10(barrad_min), np.log10(barrad_max), bar_pnts)

def lnprob(Theta): #full log-likelihood function
            
    theta = np.copy(Theta)

    #converts symmetrised to normal anisotropy (both 2nd + 4th order counterpart)
    theta[0]= 2.0*theta[0]/(1 + theta[0])

    theta[1] = 2.0*theta[1]/(1 + theta[1])

    theta[4]= 2.0*theta[4]/(1 + theta[4])

    theta[5] = 2.0*theta[5]/(1 + theta[5])
    
    for i in logthis: #exponentiate log-variables from sampling
        
        theta[i] = 10**theta[i]


    surc = theta[2*n_betpars: 2*n_betpars+nu_components] #photometric parameters


    #defines tracer densities and stellar if present mass

    if individual == True:

        surc = np.append(np.array([1.0]), surc)

    rhobs = alpbetgamden(nu_rad, *surc)
    
    sobs = alpbetgamsurf(nu_rad, *surc, bar_pnts)
        
    def baranr(r): #tracer 3D density
        
        surf = np.interp(r,nu_rad,rhobs,right = 1e-30)
        
        return surf
    
    def barsurf(r): #tracer 2D density
        
        surf = np.interp(r,nu_rad,sobs,right = 1e-30)
        
        return surf

    
    #total mass of alpha-beta-gamma profile
    rho0, r0, alp, bet, gam = surc
    normmass = 12.566370614359172 * rho0 * ((1/r0)**alp)**((-3 + gam)/alp) *  (1/r0)**(-gam) \
    * gamma((bet - gam)/alp + (-3 + gam)/alp) * gamma((3 - gam)/alp) \
/(gamma((bet - gam)/alp) * alp)


    if Mstar > 0.0:

            def maser(r, mst): #stellar profile 3D mass (assumes mass follows light)

                 return alpbetgammass(r,*surc) * mst / normmass

            Mpars = theta[2*n_betpars+nu_components:2*n_betpars+nu_components+7] #mass model parameters
    
            def M(r, Mparsu): #mass profile with stellar component
            
                Mpars = np.copy(Mparsu)
        
                M200, c, rc, n, rt, delta, Mst = Mpars #coreNFWtides params. + stellar mass
        
                return  corenfw_tides_mass(r,M200,c,rc,n,rt,delta) + maser(r, Mst)

    else:

            Mpars = theta[2*n_betpars+nu_components:2*n_betpars+nu_components+6] #mass model parameters

            def M(r, Mparsu):  #mass profile without stellar component
                
                    Mpars = np.copy(Mparsu)
            
                    M200, c, rc, n, rt, delta = Mpars #coreNFWtides params.
            
                    return  corenfw_tides_mass(r,M200,c,rc,n,rt,delta)

    betpars, betppars = theta[:4], theta[4:8]
    
    return lnlike(baranr,barsurf,M,beta,betaf, Mpars,\
                 betpars, betppars, rmin, rmax, normmass = normmass)

#Sets-up machinery for velocity PDFs based on Sanders & Evans (2020) + J. Sanders's GitHub repository

aau = np.linspace(1e-6, 100, 10**4) # 1.8 < k < 3.0 
kku = 3.0 - (2 * aau**4/15) * (1 + aau**2/3)**(-2)
def asolveu(k): #function for mapping kurtosis to uniform kernel PDF parameters

    return np.interp(k, kku[::-1], aau[::-1])  

def lukpdf(x, var, kurt, err):  #uniform kernel log-PDF for a given variance, kurtosis (1.8 < kurt < 3.0) and Gaussian error to convolve with

        sigma = var**0.5

        w = (x)/sigma
        werr = err/sigma

        a = asolveu(kurt)

        b = ((1 + a**2/3))**0.5

        t = np.sqrt(1.+b*b*werr*werr)

        delta, w0 = 0.0, 0.0
        
        am, ap = a-delta, a+delta
        it = 1./t
        bw = b*(w-w0)

        ln_pdf  = np.log(.5*b/a)+np.where((b*w+a)*it<0.,
                       logdiffexp(log_ndtr((bw+a)*it),log_ndtr((bw-a)*it)),
                        logdiffexp(log_ndtr(-(bw-a)*it),log_ndtr(-(bw+a)*it)),
                      )
        ln_pdf -= np.log(sigma)
        
        return ln_pdf

aal = np.linspace(1e-6, 100, 10**4) # 3.0 < k < 6.0
kkl = 3.0 + 12 * aal**4 * (2 * aal**2 + 1)**(-2)
def asolvel(k): #function for mapping kurtosis to Laplacian kernel PDF parameters

    return np.interp(k, kkl, aal)
   
def llkpdf(x, var, kurt, err): #Laplacian kernel log-PDF for a given variance, kurtosis (3.0 < kurt < 6.0) and Gaussian error to convolve with

    sigma = var**0.5

    w = (x)/sigma
    werr = err/sigma

    a = asolvel(kurt)

    b = (2 * a**2 + 1)**0.5

    delta, mean_w  = 0.0, 0.0

    t = np.sqrt(1.+b*b*werr*werr)
    
    ap = a+delta
    am = a-delta
    
    argU = (t*t-2*ap*b*(w-mean_w))
    positive_term = np.zeros_like(x)
    
    prefactor = np.log(b/(4.*ap))
    if type(kurt) is np.ndarray:
        prefactor = prefactor[argU<0.]
    positive_term[argU<0.] = prefactor+(argU/2./ap**2)[argU<0.]+\
                                lnerfc(((t*t-ap*b*(w-mean_w))/np.sqrt(2)/t/ap)[argU<0.])
    
    prefactor = np.log(b/ap)
    if type(kurt) is np.ndarray:
        prefactor = prefactor[argU>0.]
    positive_term[argU>0.]=.5*np.log(np.pi/8.)+prefactor+lnalpha((b*(w-mean_w)/t)[argU>0.])+\
                            lnerfcx(((t*t-ap*b*(w-mean_w))/np.sqrt(2)/t/ap)[argU>0.])
    
    argU = (t*t+2*am*b*(w-mean_w))
    negative_term = np.zeros_like(x)
    
    prefactor = np.log(b/(4.*am))
    if type(kurt) is np.ndarray:
        prefactor = prefactor[argU<0.]
    negative_term[argU<0.] = prefactor+(argU/2./am**2)[argU<0.]+\
                                lnerfc(((t*t+am*b*(w-mean_w))/np.sqrt(2)/t/am)[argU<0.])
    prefactor = np.log(b/am)
    if type(kurt) is np.ndarray:
        prefactor = prefactor[argU>0.]
    negative_term[argU>0.]=.5*np.log(np.pi/8.)+prefactor+lnalpha((b*(w-mean_w)/t)[argU>0.])+\
                            lnerfcx(((t*t+am*b*(w-mean_w))/np.sqrt(2)/t/am)[argU>0.])
    
    ln_pdf = np.logaddexp(positive_term,negative_term)-np.log(sigma)

    return ln_pdf

def lnpdfj(x, var, kurt, err): #log-PDF for a given variance, kurtosis (1.8 < kurt < 6.0) and Gaussian error to convolve with

     return np.select([kurt < 3.0, kurt >= 3.0], [lukpdf(x, var, kurt, err), llkpdf(x, var, kurt, err)])

#rLOS, vLOS, vLOS_err, rPM, vPMt, vPMt_err, vPMR, vPMR_err (are LOS and proper-motion positions and velocities with errors, 
# defined by the initialise file

#log-likelihood with proper motions (and kurtosis)
def lnlike_prop_k(nu, Sigfunc,M,beta,betaf,Mpars,\
                 betpars, betppars, rmin,rmax, normmass = 1.0):

    #remove models that cannot retain member stars (escape velocity condition)
    if esc_check == True:

        escs = escv(np.append(rLOS, rPM), Mpars, rmin, rmax, pnts = 10**3, M = M)

        for vel, esc in zip(vLOS, escs):

            if np.abs(vel) > esc:

                return -np.inf 

        for vel1, vel2, esc in zip(vPMR, vPMt, escs[len(vLOS):]):

            if np.abs(vel1) > esc or np.abs(vel2) > esc:

                return -np.inf
   
    #model photometry, velocities, and kurtoses
    Sigout, pmt2, pmr2, los2, kt, kr, kl, neg = sigp_prop_k(R,rLOS,rPM,nu,Sigfunc,M,beta,betaf,Mpars,\
                  betpars, betppars, rmin,rmax, nonn = rcn)


    #removes models with unphysical moments
    if neg == True or np.isnan(np.sum(kl)) or np.isnan(np.sum(kt)) or np.isnan(np.sum(kr)):
    
         return -np.inf

    pmt2, pmr2, los2 = pmt2/1e6, pmr2/1e6, los2/1e6
    
    lnlike_out = np.sum(lnpdfj(vLOS, los2, kl, vLOS_err)) 
    lnlike_out += np.sum(lnpdfj(vPMt, pmt2, kt, vPMt_err)) + np.sum(lnpdfj(vPMR, pmr2, kr, vPMR_err))

    if individual == True:

        lnlike_out += np.sum(np.log(R * Sigout))  - len(R) * np.log(normmass) #photometric likelihood of individual stars, no binning

    else:
    
        lnlike_out -= 0.5 * np.sum((surfden - Sigout)**2 / surfdenerr**2) #Surface brightness / number density profile likelihood

    if np.isnan(lnlike_out):
          lnlike_out = -np.inf
                    
    return lnlike_out

#log-likelihood with LOS velocities only (and kurtosis)
def lnlike_k(nu, Sigfunc,M,beta,betaf,Mpars,\
                 betpars, betppars, rmin,rmax, normmass = 1.0):

    #remove models that cannot retain member stars (escape velocity condition)
    if esc_check == True:

        escs = escv(rLOS, Mpars, rmin, rmax, pnts = 10**3, M = M)

        for vel, esc in zip(vLOS, escs):

            if np.abs(vel) > esc:

                return -np.inf 

    #model photometry, velocities, and kurtoses
    Sigout, los2, kl, neg = sigp_k(R,rLOS,nu,Sigfunc,M,beta,betaf,Mpars,\
                  betpars, betppars, rmin,rmax, nonn = rcn)
    
    los2 = los2/1e6

    #removes models with unphysical moments
    if neg==True or np.isnan(np.sum(kl)):
    
         return -np.inf

    lnlike_out = np.sum(lnpdfj(vLOS, los2, kl, vLOS_err))

    if individual == True:

        lnlike_out += np.sum(np.log(R * Sigout))  - len(R) * np.log(normmass) #photometric likelihood of individual stars, no binning

    else:
    
        lnlike_out -= 0.5 * np.sum((surfden - Sigout)**2 / surfdenerr**2) #Surface brightness / number density profile likelihood

    if np.isnan(lnlike_out):
          lnlike_out = -np.inf
                    
    return lnlike_out


if propermotion == True: #likelihood choice w or w/o proper motions 

    lnlike = lnlike_prop_k

else:
     lnlike = lnlike_k
    

