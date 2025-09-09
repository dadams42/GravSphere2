import numpy as np
from scipy.integrate import simpson as integrator
#from scipy.misc import derivative
from scipy.special import gamma
from scipy.integrate import quad, dblquad
from constants import *
from numpy import pi, log
multimode = 'normal'
from scipy.special import hyp2f1, gamma
from scipy.integrate import cumulative_trapezoid as intc

###########################################################
#For setting cosmology priors on coreNFWtides parameters.
def cosmo_cfunc(M200,h):
    #From Dutton & Maccio 2014. Requires as input masses 
    #defined in 200c system in units of Msun:
    c = 10.**(0.905 - 0.101 * (np.log10(M200*h)-12.))
    return c
    
def cosmo_cfunc_WDM(M200,h,OmegaM,rhocrit,mWDM):
    #Use formula in https://arxiv.org/pdf/1112.0330.pdf
    #to modify CDM M200-c200 relation to the WDM 
    #one. Assumes mWDM in keV, dimensionless h
    #M200 in Msun and rhocrit in Msun kpc^-3.
    cCDM = cosmo_cfunc(M200,h)
    gamma1 = 15.0
    gamma2 = 0.3
    lamfseff = 0.049*(mWDM)**(-1.11)*\
        (OmegaM/0.25)**(0.11)*(h/0.7)**(1.22)*1000.0
    lamhm = 13.93*lamfseff
    Mhm = 4.0/3.0*np.pi*rhocrit*(lamhm/2.0)**3.0
    cWDM = cCDM * (1.0 + gamma1*Mhm / M200)**(-gamma2)
    return cWDM

###########################################################
#For constraining particle DM models:
def rhoNFW(r,rhos,rs):
    return rhos/((r/rs)*(1.+(r/rs))**2.)

fac = 102

def delc(c):
    return (fac/3)*c**3/(log(1+c)-c/(1+c))

def rho_nfw(r, mvir, cvir):
    
    rv = (mvir/((4/3) * pi * fac * rhocrit))**(1/3) 
    
    rs = rv / cvir
    
    rhos = rhocrit*delc(cvir)
    
    return rhoNFW(r, rhos, rs)

def mass_nfw(r, mvir, cvir):
    
    rv = (mvir/((4/3) * pi * fac * rhocrit))**(1/3)
    
    rs = rv / cvir
    
    rhos = rhocrit*delc(cvir)
    
    return 4 * pi *  rhos * rs**3 * (log((r + rs) / rs) + rs/(r + rs) - 1) 

#rmax, vmax for NFW
def maxvals(mvir, cvir):
    
    rv = (mvir/((4/3) * np.pi * fac * rhocrit))**(1/3) 
    
    rs = rv / cvir
    
    rhos = rhocrit*delc(cvir)
    
    phis = 4 * np.pi * G * (rhos * Msun / kpc**3) * (rs * kpc)**2
    
    return np.array([2.16*rs, 0.465*np.sqrt(phis)/1000])

def nfw_dlnrhodlnr(r, mvir, cvir):
    dden = derivative(\
        lambda x: rho_nfw(x, mvir, cvir),\
        r,dx=1e-6)
    dlnrhodlnr = dden / rho_nfw(r, mvir, cvir) * r
    return dlnrhodlnr
   

def sidm_novel(rc,M200,c,oden,rhocrit):
    #Calculate SIDM model parameters from the coreNFWtides
    #model fit. For this to be valid, the coreNFWtides fit
    #should assume a pure-core model, with n=1. See
    #Read et al. 2018 for further details.
    #Returns cross section/particle mass in cm^2 / g.
    GammaX = 0.005/(1e9*year)
    Guse = G*Msun/kpc
    rho_unit = Msun/kpc**3.0
    rc = np.abs(rc)*10.0
    gcon=1./(np.log(1.+c)-c/(1.+c))
    deltachar=oden*c**3.*gcon/3.
    rv=(3./4.*M200/(np.pi*oden*rhocrit))**(1./3.)
    rs=rv/c
    rhos=rhocrit*deltachar

    rhorc = rhoNFW(rc,rhos,rs)
    r = np.logspace(np.log10(rc),np.log10(rs*5000.0),50000)
    rho = rhoNFW(r,rhos,rs)
    mass = M200*gcon*(np.log(1.0 + r/rs)-r/rs/(1.0+r/rs))
    sigvtworc = Guse/rhorc*integrator(mass*rho/r**2.0,r)
    sigvrc = np.sqrt(sigvtworc)

    sigm = np.sqrt(np.pi)*GammaX/(4.0*rhorc*rho_unit*sigvrc)
    return sigm*100.0**2.0/1000.0

def radius_dsph(s, b, distance):
    return np.sqrt((distance * np.sin(b))**2. + s*s)

def integrand(s, b, distance, rho, Mpars):
    value = np.sin(b) * rho(np.array([radius_dsph(s, b, distance)]), Mpars)**2
    return value

def integrand_D(s, b, distance, rho, Mpars):
    value = np.sin(b) * rho(np.array([radius_dsph(s, b, distance)]), Mpars)
    return value

def get_J(rho, Mpars, distance, r_max):
    """
    Compute the J factor.
    :param distance: the distance of the galaxy in kpc
    :param r_max: the maximum radius over which to integrate
                  [this gives an integration angle of
                   alpha = r_max/distance (rads)]
    :param r: the radius array for the density profile in kpc
    :param rho: the density array for the density profile in Msun/kpc^3
    :return: the J factor in in GeV c^-4 cm^-5
    """
    
    #Min/max integration angles in radians:
    b_min = 0.0
    b_max = np.arcsin(r_max/distance)
    
    #This is an appropriate choice for Dwarf galaxies but
    #should be reconsidered for large mass systems:
    Rmaximum = 250.0
    
    #Upper/lower limits:
    s_min_bound = lambda b :  -(Rmaximum**2 - (distance*np.sin(b))**2 )**0.5
    s_max_bound = lambda b : (Rmaximum**2 - (distance*np.sin(b))**2 )**0.5
    
    #Computation J_max:
    Acc_arr = 1.0e-8
    J_max = dblquad(integrand,b_min,b_max,s_min_bound,\
                    s_max_bound,args=(distance,rho,Mpars),\
                    epsabs=Acc_arr,epsrel=Acc_arr)
    J_max = J_max[0]*kpccm*2.*np.pi*Msunkpc3toGeVcm3**2.0

    #Error checking:
    if (J_max == np.inf):
        print('Argh! Infinite J_max!! Bye bye...')
        sys.exit(0)
        
    if (J_max < 0):
        print('Argh! Negative J_max!! Bye bye...')
        sys.exit(0)

    return J_max  # in GeV^2 c^-4 cm^-5

def get_D(rho, Mpars, distance, r_max):
    """
    Compute the D factor.
    :param distance: the distance of the galaxy in kpc
    :param r_max: the maximum radius over which to integrate
                  [this gives an integration angle of
                   alpha = r_max/distance (rads)]
    :param r: the radius array for the density profile in kpc
    :param rho: the density array for the density profile in Msun/kpc^3
    :return: the D factor in in GeV c^-2 cm^-2
    """

    # Min/max integration angles in radians:
    r_min = 0.0
    b_min = np.arcsin(r_min/distance)
    b_max = np.arcsin(r_max/distance)
                        
    #This is an appropriate choice for Dwarf galaxies but
    #should be reconsidered for large mass systems:
    Rmaximum = 250.0
    
    #Upper/lower limits:
    s_min_bound = lambda b :  -(Rmaximum**2 - (distance*np.sin(b))**2 )**0.5
    s_max_bound = lambda b : (Rmaximum**2 - (distance*np.sin(b))**2 )**0.5
    
    #Computation J_max:
    Acc_arr = 1.0e-8          
    D_max = dblquad(integrand_D,b_min,b_max,s_min_bound,\
                       s_max_bound,args=(distance,rho,Mpars),\
                       epsabs=Acc_arr,epsrel=Acc_arr)
    D_max = D_max[0]*kpccm*2.*np.pi*Msunkpc3toGeVcm3

    #Error checking:
    if (D_max == np.inf):
        print('Argh! Infinite D_max!! Bye bye...')
        sys.exit(0)
    if (D_max < 0):
        print('Argh! Negative D_max!! Bye bye...')
        sys.exit(0)
        
    return D_max  # in GeV c^-2 cm^-2


###########################################################
#For DM mass profile:
def corenfw_tides_den(r,M200,c,rc,n,rt,delta):
    gcon=1./(np.log(1.+c)-c/(1.+c))
    deltachar=oden*c**3.*gcon/3.
    rv=(3./4.*M200/(np.pi*oden*rhocrit))**(1./3.)
    rs=rv/c

    rhos=rhocrit*deltachar
    rhoanal = rhos/((r/rs)*(1.+(r/rs))**2.)
    manal = M200 * gcon * (np.log(1.0 + r/rs)-r/rs/(1.0+r/rs))

    x = r/np.abs(rc)
    f = np.tanh(x)
    my_manal = manal*f**n
    my_rhoanal = rhoanal*f**n + \
        1.0/(4.*np.pi*r**2.*np.abs(rc))*manal*(1.0-f**2.)*n*f**(n-1.0)
    frt = np.tanh(rt/np.abs(rc))
    manal_rt = M200 * gcon * (np.log(1.0 + rt/rs)-rt/rs/(1.0+rt/rs))
    my_rhoanal_rt = rhos/((rt/rs)*(1.+(rt/rs))**2.)*frt**n + \
        1.0/(4.*np.pi*rt**2.*np.abs(rc))*manal_rt*(1.0-frt**2.)*n*frt**(n-1.0)

    my_rhoanal[r > rt] = my_rhoanal_rt * (r[r > rt]/rt)**(-delta)

    return my_rhoanal

def corenfw_tides_mass(r,M200,c,rc,n,rt,delta):
    gcon=1./(np.log(1.+c)-c/(1.+c))
    deltachar=oden*c**3.*gcon/3.
    rv=(3./4.*M200/(np.pi*oden*rhocrit))**(1./3.)
    rs=rv/c

    rhos=rhocrit*deltachar
    rhoanal = rhos/((r/rs)*(1.+(r/rs))**2.)
    manal = M200 * gcon * (np.log(1.0 + r/rs)-r/rs/(1.0+r/rs))

    x = r/np.abs(rc)
    f = np.tanh(x)
    my_manal = manal*f**n

    frt = np.tanh(rt/np.abs(rc))
    manal_rt = M200 * gcon * (np.log(1.0 + rt/rs)-rt/rs/(1.0+rt/rs))
    my_rhoanal_rt = rhos/((rt/rs)*(1.+(rt/rs))**2.)*frt**n + \
        1.0/(4.*np.pi*rt**2.*np.abs(rc))*manal_rt*(1.0-frt**2.)*n*frt**(n-1.0)
    Mrt = manal_rt*frt**n

    my_manal[r > rt] = Mrt + \
        4.0*np.pi*my_rhoanal_rt*rt**3.0/(3.0-delta)*\
        ((r[r > rt]/rt)**(3.0-delta)-1.0)

    return my_manal

def corenfw_tides_dlnrhodlnr(r,M200,c,rc,n,rt,delta, dx = 1e-5):
    
    dden = (corenfw_tides_den(r + dx,M200,c,rc,n,rt,delta) - corenfw_tides_den(r,M200,c,rc,n,rt,delta)) / dx
    
    dlnrhodlnr = dden / corenfw_tides_den(r,M200,c,rc,n,rt,delta) * r
    
    return dlnrhodlnr
    
def vmax_func(M200,c200,h):
    oden = 200.0
    Guse = G*Msun/kpc
    r200=(3./4.*M200/(np.pi*oden*rhocrit))**(1./3.)
    
    #This from Sigad et al. 2000 (via Schneider et al. 2017):
    vmax = 0.465*np.sqrt(Guse*M200/r200)/\
           np.sqrt(1.0/c200*np.log(1.0+c200)-(1.0+c200)**(-1.0))
    return vmax/kms
    

def osipkov(r,r0):
    return r**2.0/(r**2.0+r0**2.0)
    
def gfunc_osipkov(r,r0):
    n0 = 2.0
    bet0 = 0.0
    betinf = 1.0
    gfunc = r**(2.0*betinf)*\
        ((r0/r)**n0+1.0)**(2.0/n0*(betinf-bet0))
    return gfunc

def alpbetgamvsp(rho0s,r0s,alps,bets,gams,rho0,r0,alp,bet,gam,ra):
    intpnts = int(1e4)
    r = np.logspace(np.log10(r0s/50.0),np.log10(500.0*r0s),\
                    int(intpnts))
    nu = alpbetgamden(r,rho0s,r0s,alps,bets,gams)
    massnu = alpbetgamden(r,rho0s,r0s,alps,bets,gams)
    mass = alpbetgammass(r,rho0,r0,alp,bet,gam)
    sigr = alpbetgamsigr(r,rho0s,r0s,alps,bets,gams,rho0,\
                         r0,alp,bet,gam,ra)
    bet = osipkov(r,ra)
    sigstar = np.zeros(len(r))
    for i in range(1,len(r)-3):
        sigstar[i] = 2.0*integrator(nu[i:]*r[i:]/\
                               np.sqrt(r[i:]**2.0-r[i-1]**2.0),\
                               r[i:])
 
    #Normalise similarly to the data:
    norm = integrator(sigstar*2.0*np.pi*r,r)
    nu = nu / norm
    sigstar = sigstar / norm

    #VSPs:
    vsp1 = \
        integrator(2.0/5.0*Guse*mass*nu*(5.0-2.0*bet)*\
            sigr*r,r)/1.0e12
    vsp2 = \
        integrator(4.0/35.0*Guse*mass*nu*(7.0-6.0*bet)*\
            sigr*r**3.0,r)/1.0e12
        
    #Richardson & Fairbairn zeta parameters:
    Ntotuse = integrator(sigstar*r,r)
    sigint = integrator(sigstar*r**3.0,r)
    zeta_A = 9.0/10.0*Ntotuse*integrator(Guse*mass*nu*(\
        5.0-2.0*bet)*sigr*r,r)/\
        (integrator(Guse*mass*nu*r,r))**2.0
    zeta_B = 9.0/35.0*Ntotuse**2.0*\
        integrator(Guse*mass*nu*(7.0-6.0*bet)*sigr*r**3.0,r)/\
        ((integrator(Guse*mass*nu*r,r))**2.0*sigint)
    return vsp1, vsp2, zeta_A, zeta_B

#Richardson-Fairbairn VSP estimators:
def richfair_vsp(vz,Rkin,mskin):
    vsp1_RF = 1.0/(np.pi*2.0)*\
        np.sum(vz**4.0*mskin)/np.sum(mskin)
    vsp2_RF = 1.0/(np.pi*2.0)*\
        np.sum(vz**4.0*mskin*Rkin**2.0)/np.sum(mskin*Rkin**2.0)
    return vsp1_RF, vsp2_RF


###########################################################
#For optional central dark mass (e.g. remnants, black hole):
def plumden(r,pars):
    return 3.0*pars[0]/(4.*np.pi*pars[1]**3.)*\
        (1.0+r**2./pars[1]**2.)**(-5./2.)

def plummass(r,pars):
    return pars[0]*r**3./(r**2.+pars[1]**2.)**(3./2.)


###########################################################

#alpha-beta-gamma profiles for mass models and tracers

def alpbetgamden(r,rho0,r0,alp,bet,gam):
    return rho0*(r/r0)**(-gam)*(1.0+(r/r0)**alp)**((gam-bet)/alp)

def alpbetgamdlnrhodlnr(r,rho0,r0,alp,bet,gam):
    return -gam + (gam-bet)*(r/r0)**alp*(1.0+(r/r0)**alp)**(-1.0)

def alpbetgammass(r,rho0,r0,alp,bet,gam): #exact mass profile valid for alp > 0, gam < 3
        x = r / r0
        a = (3.0 - gam) / alp
        b = (bet - gam) / alp
        c = a + 1.0
        
        # Evaluate the hypergeometric function
        hyper = hyp2f1(a, b, c, -x**alp)   
        
        # Compute M(r)
        mass = -(1/(gam - 3))*(4 * np.pi * rho0 * r0**3) * x**(3 - gam) * hyper
        
        return mass

def alpbetgamsigr(r,rho0s,r0s,alps,bets,gams,rho0,r0,alp,bet,gam,ra):
    nu = alpbetgamden(r,rho0s,r0s,alps,bets,gams)
    mass = alpbetgammass(r,rho0,r0,alp,bet,gam)
    gf = gfunc_osipkov(r,ra)
    sigr = np.zeros(len(r))
    for i in range(len(r)-3):
        sigr[i] = 1.0/nu[i]/gf[i] * \
                  integrator(Guse*mass[i:]*nu[i:]/r[i:]**2.0*\
                             gf[i:],r[i:])
    return sigr


#For improved surface brightness model (if using 2 alpha-beta-gamma components):
def doublealpbetgamden(r,rho01,rho02,r01,r02,alp1,alp2,\
                       bet1,bet2,gam1,gam2):
    return rho01*(r/r01)**(-gam1)*(1.0+(r/r01)**alp1)**((gam1-bet1)/alp1)+\
        rho02*(r/r02)**(-gam2)*(1.0+(r/r02)**alp2)**((gam2-bet2)/alp2)

def doublealpbetgammass(r,rho01,rho02,r01,r02,alp1,alp2,\
                        bet1,bet2,gam1,gam2):
    
                den = doublealpbetgamden(r,rho01,rho02,r01,r02,alp1,alp2,\
                       bet1,bet2,gam1,gam2)
                mass = np.zeros(len(r))
                for i in range(3,len(r)):
                    mass[i] = integrator(4.0*np.pi*r[:i]**2.*den[:i],r[:i])
                return mass

def doublealpbetgamsurf(r,rho01,rho02,r01,r02,alp1,alp2,\
                        bet1,bet2,gam1,gam2,intpnts):
    theta = np.linspace(0,np.pi/2.0-1.0e-30,num=intpnts)
    cth = np.cos(theta)
    cth2 = cth**2
    surf = np.zeros(len(r), 'double')
    for i in range(len(r)):
        q = r[i]/cth
        y = doublealpbetgamden(q,rho01,rho02,r01,r02,alp1,alp2,\
                               bet1,bet2,gam1,gam2)
        surf[i] = 2.0*r[i]*integrator(y/cth2,theta)
    return surf


def alpbetgamsurf(r,rho0,r0,alp,bet,gam,intpnts):
    theta = np.linspace(0,np.pi/2.0-1.0e-30,num=intpnts)
    cth = np.cos(theta)
    cth2 = cth**2
    surf = np.zeros(len(r), 'double')
    for i in range(len(r)):
        q = r[i]/cth
        y = alpbetgamden(q,rho0,r0,alp,bet,gam)
        surf[i] = 2.0*r[i]*integrator(y/cth2,theta)
    return surf

#For Jeans modelling:
#solution to 4th-order Jeans equations with PMs
#Calculates projected velocity dispersion and kurtosis profiles
#given input *functions* Sigfunc(r) (tracer surface density), nu(r) (tracer 3D density);
#mass profile M(r); anisotropy beta(r); anisotropy function betaf(r), and anisotropy + 4th-order anisotropy parameters.
def sigp_prop_k(r1,r2,r3,baranr,Sigfunc,M,beta,betaf,Mpars,\
                 betpars, betppars, rmin,rmax, intpnts = 150, nonn = False): # M is the total enclosed mass
 
    G = Guse
    #Set up theta integration array:
    intpnts = int(intpnts)
    #intpnts = int(300)
    thmin = 0.
    bit = 1.e-5
    thmax = np.pi/2.-bit
    th = np.linspace(thmin,thmax,intpnts)
    sth = np.sin(th)
    cth = np.cos(th)
    cth2 = cth**2.
    cth3 = cth**3.
    cth4 = cth**4.
    

    rint = np.logspace(np.log10(rmin),np.log10(rmax),intpnts)
    
    sigr2 = np.zeros(len(rint))
    nur = baranr(rint)
    betafunc = betaf(rint, betpars)

    #3D vel. dispersion
    for i in range(len(rint)):
        rq = rint[i]/cth
        Mq = M(rq,Mpars)
        nuq = baranr(rq)
        betafuncq = betaf(rq, betpars)
        sigr2[i] = 1./nur[i]/rint[i]/betafunc[i] * \
            integrator(G*Mq*nuq*betafuncq*sth,th)

    #calculates projected velocities along LOS and PM directions
    Sig = Sigfunc(rint)
    sigLOS2 = np.zeros(len(rint))
    sigpmr2 = np.zeros(len(rint))
    sigpmt2 = np.zeros(len(rint))
    for i in range(len(rint)):
        rq = rint[i]/cth
        nuq = baranr(rq)
        sigr2q = np.interp(rq,rint,sigr2,right=0)
        betaq = beta(rq, betpars)
        sigpmr2[i] = 2.0*rint[i]/Sig[i]*\
            integrator((1.0-betaq+betaq*cth2)*nuq*sigr2q/cth2,th)
        sigpmt2[i] = 2.0*rint[i]/Sig[i]*\
            integrator((1.0-betaq)*nuq*sigr2q/cth2,th)
        sigLOS2[i] = 2.0*rint[i]/Sig[i]*\
                     integrator((1.0-betaq*cth2)*nuq*sigr2q/cth2,th) #LOS
        
        
    #calculates fourth-order moments
    sigr4 = np.zeros(len(rint))
    betafunc = betaf(rint, betppars)
    for i in range(len(rint)):
        rq = rint[i]/cth
        sigr2q = np.interp(rq,rint,sigr2,right=0)
        Mq = M(rq,Mpars) * 3.0 * sigr2q
        # if (Mstar > 0):
        #     Mq = Mq+maser(rq, Mstar)
        nuq = baranr(rq)
        betafuncq = betaf(rq, betppars)
        sigr4[i] = 1./nur[i]/rint[i]/betafunc[i] * \
            integrator(G*Mq*nuq*betafuncq*sth,th)

    
    #projected 4th-order velocity moments for kurtosis along LOS and PM directions
    sigLOS4 = np.zeros(len(rint))
    sigpmr4 = np.zeros(len(rint))
    sigpmt4 = np.zeros(len(rint))  
    
    for i in range(len(rint)):
        rq = rint[i]/cth
        nuq = baranr(rq)
        sigr4q = np.interp(rq,rint,sigr4,right=0)
        bpq = beta(rq, betppars)
        dbpq = dbeta(rq, betppars)
        bdif = 0.75 * cth3 * \
        (bpq - beta(rq, betpars)) * np.interp(rq,rint,sigr2,right=0) \
        * G * M(rq,Mpars) / rq**2 

        sigLOS4[i] = 2.0*rint[i]/Sig[i]*\
                     integrator(((\
                      1.0 - 2.0*bpq * cth2 + 0.5*bpq*(1.0 + bpq) * cth4 - 0.25 * dbpq * rint[i] * cth3)*sigr4q\
                              + rint[i] * bdif)*nuq/cth2,th)
        
        sigpmt4[i] = 2.0*rint[i]/Sig[i]*\
                     integrator(((\
                     (1.0 - bpq)*(2 - bpq) - 0.5 * rq * dbpq)*0.5*sigr4q\
                              + rint[i] * bdif)*nuq/cth2,th)
        
        sigpmr4[i] = 2.0*rint[i]/Sig[i]*\
                     integrator((((\
                     (1.0 - bpq)*(2 - bpq) - 0.5 * rq * dbpq)*(1.0 - 2.0 * cth2 + cth4) * 0.5  \
                              + 2*(1 - bpq) * cth2 - (1 - 2 * bpq) * cth4)*sigr4q \
                                 + rint[i] * bdif * (1.0 - 2.0 * cth2 + 2.0 * cth4))*nuq/cth2,th)

    #check that there are no negative moments over the range specified
    neg = False
    
    if nonn != False: #ensure non-negative moments at r = nonn, if not, output will be rejected by likelihood

        for i in range(1, len(rint) - 1):

             if rint[-i] <= nonn:

                 break

        if np.min(np.array([sigLOS4[:-i], sigpmr4[:-i], sigpmt4[:-i]])) < 0.0:

               neg = True
            
    Sigout = np.interp(r1,rint,Sig,left=0,right=0) #photometric tracer profile
    
    sigLOS2out = np.interp(r2,rint,sigLOS2,right=0) #LOS dispersion
    sigpmr2out = np.interp(r3,rint,sigpmr2,right=0) #PM, R dispersion
    sigpmt2out = np.interp(r3,rint,sigpmt2,right=0) #PM, t dispersion
    
    sigLOS4out = np.interp(r2,rint,sigLOS4,right=0) #LOS 4th-order moment
    sigpmr4out = np.interp(r3,rint,sigpmr4,right=0) #PM, R 4th-order moment
    sigpmt4out = np.interp(r3,rint,sigpmt4,right=0) #PM, t 4th-order moment

    return Sigout, sigpmt2out, sigpmr2out, sigLOS2out,  sigpmt4out/sigpmt2out**2, sigpmr4out/sigpmr2out**2, sigLOS4out/sigLOS2out**2, neg


#same as the function above but only for LOS velocities (no PMs)
def sigp_k(r1,r2,baranr,Sigfunc,M,beta,betaf,Mpars,\
                 betpars, betppars, rmin,rmax, nonn = False): 
    G = Guse
    #Set up theta integration array:
    intpnts = int(150)
    thmin = 0.
    bit = 1.e-5
    thmax = np.pi/2.-bit
    th = np.linspace(thmin,thmax,intpnts)
    sth = np.sin(th)
    cth = np.cos(th)
    cth2 = cth**2.
    cth3 = cth**3.
    cth4 = cth**4.
    

    rint = np.logspace(np.log10(rmin),np.log10(rmax),intpnts)
    
    sigr2 = np.zeros(len(rint))
    nur = baranr(rint)
    betafunc = betaf(rint, betpars)
        
    for i in range(len(rint)):
        rq = rint[i]/cth
        Mq = M(rq,Mpars)
        nuq = baranr(rq)
        betafuncq = betaf(rq, betpars)
        sigr2[i] = 1./nur[i]/rint[i]/betafunc[i] * \
            integrator(G*Mq*nuq*betafuncq*sth,th)
        
    Sig = Sigfunc(rint)
    sigLOS2 = np.zeros(len(rint))#LOS
    sigpmr2 = np.zeros(len(rint))
    sigpmt2 = np.zeros(len(rint))
    for i in range(len(rint)):
        rq = rint[i]/cth
        nuq = baranr(rq)
        sigr2q = np.interp(rq,rint,sigr2,right=0)
        betaq = beta(rq, betpars)
        sigpmr2[i] = 2.0*rint[i]/Sig[i]*\
            integrator((1.0-betaq+betaq*cth2)*nuq*sigr2q/cth2,th)
        sigpmt2[i] = 2.0*rint[i]/Sig[i]*\
            integrator((1.0-betaq)*nuq*sigr2q/cth2,th)
        sigLOS2[i] = 2.0*rint[i]/Sig[i]*\
                     integrator((1.0-betaq*cth2)*nuq*sigr2q/cth2,th) #LOS
        
        
    

    sigr4 = np.zeros(len(rint))
    betafunc = betaf(rint, betppars)
    for i in range(len(rint)):
        rq = rint[i]/cth
        sigr2q = np.interp(rq,rint,sigr2,right=0)
        Mq = M(rq,Mpars) * 3.0 * sigr2q
        nuq = baranr(rq)
        betafuncq = betaf(rq, betppars)
        sigr4[i] = 1./nur[i]/rint[i]/betafunc[i] * \
            integrator(G*Mq*nuq*betafuncq*sth,th)

    sigLOS4 = np.zeros(len(rint))#LOS
    sigpmr4 = np.zeros(len(rint))
    sigpmt4 = np.zeros(len(rint))
    
    for i in range(len(rint)):
        rq = rint[i]/cth
        nuq = baranr(rq)
        sigr4q = np.interp(rq,rint,sigr4,right=0)
        bpq = beta(rq, betppars)
        dbpq = dbeta(rq, betppars)
        bdif = 0.75 * cth3 * \
        (bpq - beta(rq, betpars)) * np.interp(rq,rint,sigr2,right=0) \
        * G * M(rq,Mpars) / rq**2 

        sigLOS4[i] = 2.0*rint[i]/Sig[i]*\
                     integrator(((\
                      1.0 - 2.0*bpq * cth2 + 0.5*bpq*(1.0 + bpq) * cth4 - 0.25 * dbpq * rint[i] * cth3)*sigr4q\
                              + rint[i] * bdif)*nuq/cth2,th)

    neg = False


    if nonn != False:

        for i in range(1, len(rint) - 1):

             if rint[-i] <= nonn:

                 break

    if np.min(np.array([sigLOS4[:-i]])) < 0.0:

               neg = True
        
       

    Sigout = np.interp(r1,rint,Sig,left=0,right=0) #photometric tracer profile
    
    sigLOS2out = np.interp(r2,rint,sigLOS2,right=0) #LOS dispersion
    
    sigLOS4out = np.interp(r2,rint,sigLOS4,right=0) #LOS 4th-order moment
 

    return Sigout, sigLOS2out, sigLOS4out/sigLOS2out**2, neg


def beta(r, betpars):
    
    rt = betpars[2] 
    n = betpars[3]

    bet0 = betpars[0]
    betinf = betpars[1]

    beta = bet0 + (betinf-bet0)*(1.0/(1.0 + (rt/r)**n))
    return beta

def dbeta(r, betpars):
    
    rt = betpars[2] 
    n = betpars[3]

    bet0 = betpars[0]
    betinf = betpars[1]

    dbeta = n * rt**n * r**(-(1.0 + n)) * (betinf-bet0)*(1.0/(1.0 + (rt/r)**n)**2)
    return dbeta

def betaf(r,betpars):
    
    rt = betpars[2] 
    n = betpars[3]

    bet0 = betpars[0]
    betinf = betpars[1]

    betafn = r**(2.0*betinf)*((rt/r)**n+1.0)**(2.0/n*(betinf-bet0))
    
    return betafn


def accelminmax_calc(r,M,Mcentral,Mpars,\
                     Mstar_rad,Mstar_prof,Mstar,G,rmin,rmax):

    #Calculate acceleration as a function of radius (SI units):
    intpnts = int(150)
    rq = np.logspace(np.log10(rmin),np.log10(rmax),intpnts)
    Mq = M(rq,Mpars)+Mcentral(rq,Mpars)
    if (Mstar > 0):
        Mq = Mq+Mstar*np.interp(rq,Mstar_rad,Mstar_prof)
    aq = G*Mq/rq**2.0/kpc

    #Now at each projected radius, scan alone line
    #of sight to cacl min/max acceleration:
    Rproj = r
    thmin = 0.
    bit = 1.e-5
    thmax = np.pi/2.-bit
    th = np.linspace(thmin,thmax,intpnts)
    sth = np.sin(th)
    cth = np.cos(th)
    aminmax = np.zeros(len(Rproj))
    for i in range(len(Rproj)):
        #Define angle th s.t. th = pi points towards us
        #and th = -pi points away. Acc as a function of
        #theta is: alos = np.abs(aq(Rproj[i]/cth)*sth)
        #By symmetry, only need to consider th=[0,pi/2]:
        aqq = np.interp(Rproj[i]/cth,rq,aq)
        alos = np.abs(aqq*sth)

        #The above (appropriately normalised) is the
        #likelihood of a line of sight aceleration.
        #Here, we want min/max:
        aminmax[i] = np.max(alos)
    return aminmax

###########################################################
#For data binning:
def binthedata(R,ms,Nbin):
    #Nbin is the number of particles / bin:
    index = np.argsort(R)
    right_bin_edge = np.zeros(len(R))
    norm = np.zeros(len(R))
    cnt = 0
    jsum = 0

    for i in range(len(R)):
        if (jsum < Nbin):
            norm[cnt] = norm[cnt] + ms[index[i]]
            right_bin_edge[cnt] = R[index[i]]
            jsum = jsum + ms[index[i]]
        if (jsum >= Nbin):
            jsum = 0.0
            cnt = cnt + 1
    
    right_bin_edge = right_bin_edge[:cnt]
    norm = norm[:cnt]
    surfden = np.zeros(cnt)
    rbin = np.zeros(cnt)
    
    for i in range(len(rbin)):
        if (i == 0):
            surfden[i] = norm[i] / \
                (np.pi*right_bin_edge[i]**2.0)
            rbin[i] = right_bin_edge[i]/2.0
        else:
            surfden[i] = norm[i] / \
                (np.pi*right_bin_edge[i]**2.0-\
                 np.pi*right_bin_edge[i-1]**2.0)
            rbin[i] = (right_bin_edge[i]+right_bin_edge[i-1])/2.0
    surfdenerr = surfden / np.sqrt(Nbin)
    
    #Calculate the projected half light radius &
    #surface density integral:
    Rhalf, Menc_tot = surf_renorm(rbin,surfden)

    #And normalise the profile:
    surfden = surfden / Menc_tot
    surfdenerr = surfdenerr / Menc_tot

    return rbin, surfden, surfdenerr, Rhalf

def surf_renorm(rbin,surfden):
    #Calculate the integral of the surface density
    #so that it can then be renormalised.
    #Calcualte also Rhalf.
    ranal = np.linspace(0,10,int(5000))
    surfden_ranal = np.interp(ranal,rbin,surfden,left=0,right=0)
    Menc_tot = 2.0*np.pi*integrator(surfden_ranal*ranal,ranal)
    Menc_half = 0.0
    i = 3
    while (Menc_half < Menc_tot/2.0):
        Menc_half = 2.0*np.pi*\
            integrator(surfden_ranal[:i]*ranal[:i],ranal[:i])
        i = i + 1
    Rhalf = ranal[i-1]
    return Rhalf, Menc_tot


###########################################################
#For calculating confidence intervals: 
def calcmedquartnine(array):
    index = np.argsort(array,axis=0)
    median = array[index[int(len(array)/2.)]]
    sixlowi = int(16./100. * len(array))
    sixhighi = int(84./100. * len(array))
    ninelowi = int(2.5/100. * len(array))
    ninehighi = int(97.5/100. * len(array))
    nineninelowi = int(0.15/100. * len(array))
    nineninehighi = int(99.85/100. * len(array))

    sixhigh = array[index[sixhighi]]
    sixlow = array[index[sixlowi]]
    ninehigh = array[index[ninehighi]]
    ninelow = array[index[ninelowi]]
    nineninehigh = array[index[nineninehighi]]
    nineninelow = array[index[nineninelowi]]

    return median, sixlow, sixhigh, ninelow, ninehigh,\
        nineninelow, nineninehigh


###########################################################
#For fitting the surface brightness:
def Sig_addpnts(x,y,yerr):
    #If using neg. Plummer component, add some more
    #"data points" at large & small radii bounded on
    #zero and the outermost data point. This
    #will disfavour models with globally
    #negative tracer density.
    addpnts = len(x)
    xouter = np.max(x)
    youter = np.min(y)
    xinner = np.min(x)
    yinner = np.max(y)
    xadd_right = np.logspace(np.log10(xouter),\
                             np.log10(xouter*1000),addpnts)
    yadd_right = np.zeros(addpnts) + youter/2.0
    yerradd_right = yadd_right
    xadd_left = np.logspace(np.log10(xinner),\
                            np.log10(xinner/1000),addpnts)
    yadd_left = np.zeros(addpnts) + yinner
    yerradd_left = yadd_left/2.0
    x = np.concatenate((x,xadd_right))
    y = np.concatenate((y,yadd_right))
    yerr = np.concatenate((yerr,yerradd_right))
    x = np.concatenate((xadd_left,x))
    y = np.concatenate((yadd_left,y))
    yerr = np.concatenate((yerradd_left,yerr))
    return x,y,yerr

#For stellar and tracer profiles:
def multiplumden(r,pars):
    Mpars = pars[0:int(len(pars)/2.0)]
    apars = pars[int(len(pars)/2.0):len(pars)]
    nplum = len(Mpars)
    multplum = np.zeros(len(r))
    for i in range(len(Mpars)):
        if (multimode == 'seq'):
            if (i == 0):
                aparsu = apars[0]
            else:
                aparsu = apars[i] + apars[i-1]
        else:
            aparsu = apars[i]
        multplum = multplum + \
            3.0*Mpars[i]/(4.*np.pi*aparsu**3.)*\
            (1.0+r**2./aparsu**2.)**(-5./2.)
    return multplum

def multiplumsurf(r,pars):
    Mpars = pars[0:int(len(pars)/2.0)]
    apars = pars[int(len(pars)/2.0):len(pars)]
    nplum = len(Mpars)
    multplum = np.zeros(len(r))
    for i in range(len(Mpars)):
        if (multimode == 'seq'):
            if (i == 0):
                aparsu = apars[0]
            else:
                aparsu = apars[i] + apars[i-1]
        else:
            aparsu = apars[i]
        multplum = multplum + \
            Mpars[i]*aparsu**2.0 / \
            (np.pi*(aparsu**2.0+r**2.0)**2.0)
    return multplum

def multiplumdlnrhodlnr(r,pars):
    Mpars = pars[0:int(len(pars)/2.0)]
    apars = pars[int(len(pars)/2.0):len(pars)]
    nplum = len(Mpars)
    multplumden = np.zeros(len(r))
    multplumdden = np.zeros(len(r))
    for i in range(len(Mpars)):
        if (multimode == 'seq'):
            if (i == 0):
                aparsu = apars[0]
            else:
                aparsu = apars[i] + apars[i-1]
        else:
            aparsu = apars[i]
        multplumden = multplumden + \
            3.0*Mpars[i]/(4.*np.pi*aparsu**3.)*\
            (1.0+r**2./aparsu**2.)**(-5./2.)
        multplumdden = multplumdden - \
            15.0*Mpars[i]/(4.*np.pi*aparsu**3.)*\
            r/aparsu**2.*(1.0+r**2./aparsu**2.)**(-7./2.)
    return multplumdden*r/multplumden

def multiplummass(r,pars):
    Mpars = pars[0:int(len(pars)/2.0)]
    apars = pars[int(len(pars)/2.0):len(pars)]
    nplum = len(Mpars)
    multplum = np.zeros(len(r))
    for i in range(len(Mpars)):
        if (multimode == 'seq'):
            if (i == 0):
                aparsu = apars[0]
            else:
                aparsu = apars[i] + apars[i-1]
        else:
            aparsu = apars[i]
        multplum = multplum + \
            Mpars[i]*r**3./(r**2.+aparsu**2.)**(3./2.)
    return multplum

def threeplumsurf(r,M1,M2,M3,a1,a2,a3):
    return multiplumsurf(r,[M1,M2,M3,\
                            a1,a2,a3])
def threeplumden(r,M1,M2,M3,a1,a2,a3):
    return multiplumden(r,[M1,M2,M3,\
                           a1,a2,a3])
def threeplummass(r,M1,M2,M3,a1,a2,a3):
    return multiplummass(r,[M1,M2,M3,\
                            a1,a2,a3])

def Rhalf_func(M1,M2,M3,a1,a2,a3):
    #Calculate projected half light radius for
    #the threeplum model:
    ranal = np.logspace(-3,1,int(500))
    Mstar_surf = threeplumsurf(ranal,M1,M2,M3,a1,a2,a3)

    Menc_half = 0.0
    i = 3
    while (Menc_half < (M1+M2+M3)/2.0):
        Menc_half = 2.0*np.pi*\
            integrator(Mstar_surf[:i]*ranal[:i],ranal[:i])
        i = i + 1
    Rhalf = ranal[i-1]
    return Rhalf


###########################################################
#For fitting the velocity distribution in each bin [no errors]:
def monte(func,a,b,n):
    #Function to perform fast 1D Monte-Carlo integration
    #for convolution integrals:
    xrand = np.random.uniform(a,b,n)
    integral = func(xrand).sum()
    return (b-a)/np.float(n)*integral

def velpdf_noerr(vz,theta):
    vzmean = theta[0]
    alp = theta[1]
    bet = theta[2]
    backamp = theta[3]
    backmean = theta[4]
    backsig = theta[5]

    pdf = (1.0-backamp)*bet/(2.0*alp*gamma(1.0/bet))*\
        np.exp(-(np.abs(vz-vzmean)/alp)**bet) + \
        backamp/(np.sqrt(2.0*np.pi)*backsig)*\
        np.exp(-0.5*(vz-backmean)**2.0/backsig**2.0)
    return pdf

#For fitting the velocity distribution in each bin [fast]
#Uses an approximation to the true convolution integral.
def velpdffast(vz,vzerr,theta):
    vzmean = theta[0]
    bet = theta[2]
    fgamma = gamma(1.0/bet)/gamma(3.0/bet)
    alp = np.sqrt(theta[1]**2.0+vzerr**2.0*fgamma)    
    backamp = theta[3]
    backmean = theta[4]
    backsig = np.sqrt(theta[5]**2.0 + vzerr**2.0)
    
    pdf = (1.0-backamp)*bet/(2.0*alp*gamma(1.0/bet))*\
        np.exp(-(np.abs(vz-vzmean)/alp)**bet) + \
        backamp/(np.sqrt(2.0*np.pi)*backsig)*\
        np.exp(-0.5*(vz-backmean)**2.0/backsig**2.0)
    return pdf
    
def velpdf_func(vz,vzerr,vzint,theta):
    #Inner integral function for convolving
    #velpdf with a Gaussian error PDF. Change
    #this function to implement non-Gaussian
    #errors.
    vzmean = theta[0]
    alp = theta[1]
    bet = theta[2]
    backamp = theta[3]
    backmean = theta[4]
    backsig = theta[5]
    pdf = (1.0-backamp)*bet/(2.0*alp*gamma(1.0/bet))*\
                np.exp(-(np.abs(vzint-vzmean)/alp)**bet)*\
                1.0/(np.sqrt(2.0*np.pi)*vzerr)*\
                np.exp(-0.5*(vz-vzint)**2.0/vzerr**2.0)+\
                backamp/(np.sqrt(2.0*np.pi)*backsig)*\
                np.exp(-0.5*(vzint-backmean)**2.0/backsig**2.0)*\
                1.0/(np.sqrt(2.0*np.pi)*vzerr)*\
                np.exp(-0.5*(vz-vzint)**2.0/vzerr**2.0)
    return pdf

#For fitting the velocity distribution in each bin with
#full (expensive) convolution integral:
def velpdf(vz,vzerr,theta):
    #Generalised Gaussian + Gaussian convolved with
    #vzerr, assuming Gaussian errors:
    vzmean = theta[0]
    sig = vztwo_calc(theta)
    vzlow = -sig*10+vzmean
    vzhigh = sig*10+vzmean
    if (type(vz) == np.ndarray):
        pdf = np.zeros(len(vz))
        for i in range(len(vz)):
            pdf_func = lambda vzint : velpdf_func(vz[i],\
                vzerr[i],vzint,theta)
            pdf[i] = quad(pdf_func,vzlow,vzhigh)[0]
    else:
        pdf_func = lambda vzint : velpdf_func(vz,\
            vzerr,vzint,theta)
        pdf = quad(pdf_func,vzlow,vzhigh)[0]
    return pdf

def velpdfmonte(vz,vzerr,theta):
    #Generalised Gaussian + Gaussian convolved with
    #vzerr, assuming Gaussian errors:
    npnts = int(500)
    vzmean = theta[0]
    sig = vztwo_calc(theta)
    vzlow = -sig*10+vzmean
    vzhigh = sig*10+vzmean
    if (type(vz) == np.ndarray):
        pdf = np.zeros(len(vz))
        for i in range(len(vz)):
            pdf_func = lambda vzint : velpdf_func(vz[i],\
                vzerr[i],vzint,theta)
            pdf[i] = monte(pdf_func,vzlow,vzhigh,npnts)
    else:
        pdf_func = lambda vzint : velpdf_func(vz,\
            vzerr,vzint,theta)
        pdf = monte(pdf_func,vzlow,vzhigh,npnts)
    return pdf

def vztwo_calc(theta):
    #Calculate <vlos^2>^(1/2) from
    #generalised Gaussian parameters:
    alp = theta[1]
    bet = theta[2]
    return np.sqrt(alp**2.0*gamma(3.0/bet)/gamma(1.0/bet))
    
def vzfour_calc(theta):
    #Calculate <vlos^4> from
    #generalised Gaussian parameters:
    alp = theta[1]
    bet = theta[2]
    sig = vztwo_calc(theta)
    kurt = gamma(5.0/bet)*gamma(1.0/bet)/(gamma(3.0/bet))**2.0
    return kurt*sig**4.0
    
def kurt_calc(theta):
    #Calculate kurtosis from generalised
    #Gaussian parameters:
    alp = theta[1]
    bet = theta[2]
    kurt = gamma(5.0/bet)*gamma(1.0/bet)/(gamma(3.0/bet))**2.0
    return kurt

def vzfourfunc(ranal,rbin,vzfourbin):
    #Interpolate and extrapolate
    #vzfour(R) over and beyond the data:
    vzfour = np.interp(ranal,rbin,vzfourbin)
    return vzfour
    
#For calculating the Likelihood from the vsp array:
def vsppdf_calc(vsp):
    #First bin the data:
    nbins = 50
    bins_plus_one = np.linspace(np.min(vsp),np.max(vsp),nbins+1)
    bins = np.linspace(np.min(vsp),np.max(vsp),nbins)
    vsp_pdf, bins_plus_one = np.histogram(vsp, bins=bins_plus_one)
    vsp_pdf = vsp_pdf / np.max(vsp_pdf)
    binsout = bins[vsp_pdf > 0]
    vsp_pdfout = vsp_pdf[vsp_pdf > 0]
    return binsout, vsp_pdfout

def vsp_pdf(vsp,bins,vsp_pdf):
    return np.interp(vsp,bins,vsp_pdf,left=0,right=0)


#escape velocity profile 
def escv(r, Mpars, rmin, rmax, pnts = 10**3, M = corenfw_tides_mass):

    rint = np.linspace(rmin, rmax, pnts)

    potn = intc(M(rint, Mpars) / rint**2, rint)

    pot = potn[-1] - potn

    return np.interp(r, rint[1:],(2 * pot * Guse)**0.5/1e3)

