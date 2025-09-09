# GravSphere2

GravSphere 2 is an improved solver of the spherical higher-order Jeans equations, designed to mass-model galaxies and star clusters with minimal bias and maximizing information content. It improves on previous versions of GravSphere and other methods by removing the need for binning and performing a full general treatment of higher moments. 

Here is some important information about GravSphere2 (for more information, check also python source files of the code):

:::: DEPENDENCIES ::::
To run these codes, you will need to install:
python3.x, numpy, scipy, matplotlib and dynesty

The code has been verified to work with:
python 3.10.12
scipy version 1.15.1
numpy version 1.26.4
matplotlib version 3.9.0
dynesty version 2.1.5


:::: CITATIONS ::::
If using this code, please cite the following code papers:

Bañares-Hernández et al. (release paper)

and former GravSphere papers

https://ui.adsabs.harvard.edu/abs/2017MNRAS.471.4541R/abstract
https://ui.adsabs.harvard.edu/abs/2018MNRAS.481..860R/abstract
https://ui.adsabs.harvard.edu/abs/2020MNRAS.498..144G/abstract
https://ui.adsabs.harvard.edu/abs/2021arXiv210211890C/abstract
https://ui.adsabs.harvard.edu/abs/2025A%26A...693A.104B/

You should also cite the use of dynesty for the sampling method:
https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S/abstract
And use the sampler.citations option for all the related papers
(https://dynesty.readthedocs.io/en/latest/references.html)

If using the J-factor integral calculation, please cite also:
https://ui.adsabs.harvard.edu/abs/2020JCAP...09..004A/abstract

Please acknowledge use of the gravsphere code and/or
binulator, and link to its github page:
https://github.com/justinread/gravsphere

If using the Gaia Challenge mock data, please cite:
https://ui.adsabs.harvard.edu/abs/2021MNRAS.501..978R/abstract
http://astrowiki.ph.surrey.ac.uk/dokuwiki/doku.php?id=start

If using the simulated dwarf galaxy data, please cite:
https://ui.adsabs.harvard.edu/abs/2025arXiv250418617T/abstract


:::: BUGS ::::
If you spot any bugs, please let us know!


:::: NOTES ::::
Note that this public release of the code is distinct from
the previous versions of GravSphere that don't use higher-order
Jeans or individual velocities (not 'bin free').

Former versions include
GravSphere + binulator (https://github.com/justinread/gravsphere)
which uses binning and Virial Shape Parameters
and PyGravSphere, which has non-parametric 
mass modeling and the original binning  methodology,
available at Anna Genina's 
independent public release here: 
https://github.com/AnnaGenina/pyGravSphere
new

If you are not familiar with dynesty, check its website: dynesty.readthedocs.io 

:::: EXAMPLES ::::
We have included the examples addressed in Bañares-Hernández et al.
featuring a Fornax-like simulated galaxy and the Gaia Challenge mocks

To run an example: 

1. By default, the Output directory is set in the same file, if 
   you want to change this, this can be modified in the initialise_ files,
   where you specify the data for each galaxy. Here you also decide whether
   to fit proper motions, include a massive stellar component, 'bin free' 
   fitting for the photometric tracers (not just kinematic ones) etc.

2. The galaxy / object is selected by importing the initialisation file at 
   the beginning of the gravsphere2.py file, where the likelihood and other
   arguments for the fit (e.g. priors) are specified.

3. You will need to set up folders inside your output directory
   to store the output files. We have done this for the case of
   the Fornax simulation and Gaia Challenge data as an example

4. We have set some notebooks to show an example of running the code
  (run_example.ipynb and run_multcpu_example.py for parallel programming
   with a multi-CPU machine / cluster). This will create and store a 
   checkpoint file from which all data can be retrieved and be 
   run subsequently if needed.

5. We have also created an notebook (plots_example_notebook.ipynb) where we 
   show examples of plotting density, slope, anisotropy, dispersion, and 
   kurtosis profiles, similar to those from Bañares-Hernández et al.


:::: Setting up your own model ::::

To input your own data, you should create your own initiliase_ file with
the same format as the ones from the examples. This will include tracer
positions and velocities, data for the tracer profiles (positions, which 
can be done without binning, or a binned profile if only that is available),
and information for setting up the priors. You can also change the gravsphere2
file if you e.g. want a different mass model, introduce new parameters etc.

Andrés Bañares Hernández & Justin Read| 01/08/25 


