#runs parallelized version of dynesty sampler in multi-CPU machine

###########################################################
from gravsphere2 import *
###########################################################

from multiprocessing import Pool
import sys
import dynesty
import os

print('Running dynesty ... ')

fsname = str(diro + 'Sampler Chains/fornax_pm100_example_chk')

if __name__ == '__main__':
  #ncpus = int(sys.argv[1])
  ncpus = 42 #number of CPUs, arbitrary number (depending on availability)
  pool = Pool(ncpus)
  dns = dynesty.DynamicNestedSampler(
      lnprob, ptform, ndims, pool=pool, queue_size = ncpus, nlive = 500)
  dns.run_nested(checkpoint_file=fsname, use_stop = True) 

#For running existing checkpoint file, use the code below instead
# if __name__ == '__main__':
#   ncpus = 42 #number of CPUs, arbitrary number (depending on availability)
#   pool = Pool(ncpus) 
#   dns = DynamicNestedSampler.restore(fsname, pool = pool)
#   dns.run_nested(checkpoint_file=fsname, use_stop = False, resume = True)
