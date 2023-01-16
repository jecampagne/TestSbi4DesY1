from pathinit import *    # setup  CCIN2P3 env

import os
from astropy.io import fits

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.distributions as torch_dist

from sbi.inference import SNPE, SNLE, SNRE, prepare_for_sbi, simulate_for_sbi
from sbi.simulators.simutils import simulate_in_batches
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils

import pyro
import pyro.distributions as pyro_dist

# again, this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as numpyro_dist
from numpyro.handlers import seed, trace, condition


import jax_cosmo as jc


import pickle

##
## experimetal jax - pytorch interoperability : Not necessary
## from torch.utils import dlpack as torch_dlpack
## from jax import dlpack as jax_dlpack

## def j2t(x_jax):
##     x_torch = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(x_jax))
##     return x_torch

## def t2j(x_torch):
##     x_torch = x_torch.contiguous() # https://github.com/google/jax/issues/8082
##     x_jax = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x_torch))
##     return x_jax




####
## device = "cuda:0" if torch.cuda.is_available() else "cpu"

## JEC using GPU for SBI does not produce any gain according to SBI authors
##     but running on GPU helps JAX-based simulation
device = "cpu"
print("running on device:", device)


####
##  utils
###

from  utils import *


output_dir = "./run_SNPE_mdn_cpx/"
print("saving in dir:", output_dir)

def set_seed(x):
    pyro.set_rng_seed(x)


####
# load an observation
####


nz_source=fits.getdata('2pt_NG_mcal_1110.fits', 6)
nz_lens=fits.getdata('2pt_NG_mcal_1110.fits', 7)

# This is the effective number of sources from the cosmic shear paper
neff_s = [1.47, 1.46, 1.50, 0.73]
nzs_s = [jc.redshift.kde_nz(nz_source['Z_MID'].astype('float32'),
                            nz_source['BIN%d'%i].astype('float32'), 
                            bw=0.01,
                            gals_per_arcmin2=neff_s[i-1])
           for i in range(1,5)]
nzs_l = [jc.redshift.kde_nz(nz_lens['Z_MID'].astype('float32'),
                            nz_lens['BIN%d'%i].astype('float32'), bw=0.01)
        for i in range(1,6)]


# Define some ell range
ell = np.logspace(1, 3)


# the fiducial parameters
with open('fiducial_dictionary.pkl', 'rb') as f:
    fidu_params = pickle.load(f)

print("fidu_params: ",fidu_params)

fidu_vals  = [x for x in fidu_params.values()]

# Precision matrix 
P = np.load("precision_mtx.npy")
# For later jasx/torch usage
P_jax    = jnp.asarray(P)
P_tensor = torch.tensor(P)

#JEC 13/1/23 use compressed observable
#obs = np.load("obs_xo.npy")
obs = np.load("obs_xo_comp.npy")
n_obs = len(obs)
# This is our fake data vector
fig = plt.figure()
plt.semilogy(obs);
#plt.ylabel("Cls")
plt.ylabel("obs")
plt.savefig(output_dir+"obs_xcheck.pdf")
plt.close()

# transform the data adapt the dim for sbi
x_o = torch.tensor(obs)[None]

####
# Parameters, Priors, Simulator
####


#from sbi.utils import process_prior

# Cosmological params
prior_oc = utils.BoxUniform(low=0.1 *torch.ones(1), high=0.9   *torch.ones(1),  device="cpu")
#alternative prior_oc = torch_dist.Uniform(0.1*torch.ones(1,device="cpu"),0.9*torch.ones(1, device="cpu")) 
prior_s8 = utils.BoxUniform(low=0.4 *torch.ones(1), high=1.0   *torch.ones(1),  device="cpu")
prior_ob = utils.BoxUniform(low=0.03 *torch.ones(1), high=0.07   *torch.ones(1),  device="cpu")
prior_h  = utils.BoxUniform(low=0.55*torch.ones(1), high=0.91  *torch.ones(1),  device="cpu")
prior_ns  = utils.BoxUniform(low=0.87*torch.ones(1), high=1.07  *torch.ones(1),  device="cpu")
prior_w0 = utils.BoxUniform(low=-2.0*torch.ones(1), high=-0.33 *torch.ones(1),  device="cpu")
# Intrinsic Alignment
prior_A = utils.BoxUniform(low=-5.0*torch.ones(1), high=5.0 *torch.ones(1),  device="cpu")
prior_eta = utils.BoxUniform(low=-5.0*torch.ones(1), high=5.0 *torch.ones(1),  device="cpu")
# linear galaxy bias
prior_b1 = utils.BoxUniform(low=0.8*torch.ones(1), high=3.0 *torch.ones(1),  device="cpu")
prior_b2 = utils.BoxUniform(low=0.8*torch.ones(1), high=3.0 *torch.ones(1),  device="cpu")
prior_b3 = utils.BoxUniform(low=0.8*torch.ones(1), high=3.0 *torch.ones(1),  device="cpu")
prior_b4 = utils.BoxUniform(low=0.8*torch.ones(1), high=3.0 *torch.ones(1),  device="cpu")
prior_b5 = utils.BoxUniform(low=0.8*torch.ones(1), high=3.0 *torch.ones(1),  device="cpu")
# parameters for systematics
prior_m1 = torch_dist.Normal(loc=0.012 * torch.ones(1,device="cpu"), scale=0.023 * torch.ones(1,device="cpu"))
prior_m2 = torch_dist.Normal(loc=0.012 * torch.ones(1,device="cpu"), scale=0.023 * torch.ones(1,device="cpu"))
prior_m3 = torch_dist.Normal(loc=0.012 * torch.ones(1,device="cpu"), scale=0.023 * torch.ones(1,device="cpu"))
prior_m4 = torch_dist.Normal(loc=0.012 * torch.ones(1,device="cpu"), scale=0.023 * torch.ones(1,device="cpu"))

prior_dz1 = torch_dist.Normal(loc=0.001 * torch.ones(1,device="cpu"), scale=0.016 * torch.ones(1,device="cpu"))
prior_dz2 = torch_dist.Normal(loc=-0.019 * torch.ones(1,device="cpu"), scale=0.013 * torch.ones(1,device="cpu"))
prior_dz3 = torch_dist.Normal(loc=0.009 * torch.ones(1,device="cpu"), scale=0.011 * torch.ones(1,device="cpu"))
prior_dz4 = torch_dist.Normal(loc=-0.018 * torch.ones(1,device="cpu"), scale=0.022 * torch.ones(1,device="cpu"))


prior = [prior_oc, prior_s8, prior_w0, prior_h, prior_ob, prior_ns,\
         prior_A, prior_eta,\
         prior_b1, prior_b2, prior_b3, prior_b4, prior_b5,\
         prior_m1, prior_m2, prior_m3, prior_m4, \
         prior_dz1, prior_dz2, prior_dz3, prior_dz4]

theta_dim = len(prior)



true_theta = np.array([fidu_params["Omega_c"],fidu_params["sigma8"], fidu_params["w0"], fidu_params["h"],
                       fidu_params["Omega_b"], fidu_params["n_s"],
                       fidu_params["A"], fidu_params["eta"],
                       fidu_params["b1"], fidu_params["b2"], fidu_params["b3"], fidu_params["b4"],fidu_params["b5"],
                       fidu_params["m1"], fidu_params["m2"], fidu_params["m3"], fidu_params["m4"],
                       fidu_params["dz1"], fidu_params["dz2"], fidu_params["dz3"],  fidu_params["dz4"]])

# The simulator

#JEC 13/1/23
moped_mtx = np.load("moped_vect.npy")
moped_mtx = torch.tensor(moped_mtx)

def unpack_params_vec(theta): # theta is a torch vector
  # Retrieve cosmology

  #JEC 9/11/22
  theta = theta.cpu().numpy()  # torch to numpy
  #theta = theta.to(torch.float64)
  #theta = t2j(theta)
  
  Omega_c, sigma8, w0, h, Omega_b, n_s = theta[:6]
  

  #Comoslogical parameters
  cosmo = jc.Cosmology(Omega_c=Omega_c, sigma8=sigma8,  w0=w0, 
                       Omega_b=Omega_b, h=h,n_s=n_s,
                       Omega_k=0.0,wa=0.0)

  # Intrinsic Alignment parameters
  A = theta[6]
  eta = theta[7]
  # linear galaxy bias
  bias = theta[8:13]
  # parameters for systematics
  m1,m2,m3,m4 = theta[13:17]         
  dz1,dz2,dz3,dz4 = theta[17:21]
  
  return cosmo, [m1,m2,m3,m4], [dz1,dz2,dz3,dz4], [A, eta], bias




@jax.jit
def _cmp_cl(cosmo, m, dz, A, eta, bias):
  # Build source nz with redshift systematic bias
  nzs_s_sys = [jc.redshift.systematic_shift(nzi, dzi, zmax=2.0) 
                for nzi, dzi in zip(nzs_s, dz)]
    
  # Define IA model, z0 is fixed
  b_ia = jc.bias.des_y1_ia_bias(A, eta, 0.62)
  # Bias for the lenses
  b = [jc.bias.constant_linear_bias(bi) for bi in bias] 
    
  # Define the lensing and number counts probe
  probes = [jc.probes.WeakLensing(nzs_s_sys, 
                                    ia_bias=b_ia,
                                    multiplicative_bias=m),
             jc.probes.NumberCounts(nzs_l, b)]

  cl = jc.angular_cl.angular_cl(cosmo, ell, probes).flatten()

  return cl


  
def _simulator(theta):  # theta is a torch tensor
  cosmo, m, dz, (A, eta), bias = unpack_params_vec(theta) 
  # Now that params are defined, here is the forward model

  cl = _cmp_cl(cosmo, m, dz, A, eta, bias)

  #JEC 13/1/23
  cl_tensor= torch.tensor(np.asarray(cl))  
  x = pyro_dist.MultivariateNormal(loc=cl_tensor,precision_matrix=P_tensor).sample()
  #x = numpyro_dist.MultivariateNormal(loc=cl,precision_matrix=P_jax).sample()
  x =  moped_mtx @ x

  
  return x.to(device="cpu") # cpu for SNPE/SNLE at CC 24oct22

#adapt/check the prior & simulator for SBI
simulator, prior = prepare_for_sbi(_simulator, prior)


#for ploting
keys = ["Omega_c","sigma8","w0","h"]
#        , "Omega_b", "n_s",
#        "A", "eta",
#        "b1","b2","b3","b4","b5",
#        "m1","m2","m3","m4",
#        "dz1","dz2","dz3","dz4"]

truth = dict(zip(keys,true_theta[:4]))

####
# Bagging optimisation of methods in SNPE-C
####

# SNPE : posterior_nn
from sbi.utils.get_nn_models import posterior_nn





def do_multi_pass(num_simu = 10_000, num_rounds = 5, tag="default", max_num_epochs=None, do_plot=True):




    # multi rounds: first round simulates from the prior, second round simulates parameter set
    # that were sampled from the obtained posterior.
    # The specific observation we want to focus the inference on is x_o (single)


    if max_num_epochs is None:
        max_num_epochs = 2**31 - 1


    # maf default:  hidden_features: int = 50, num_transforms: int = 5,  num_blocks: int = 2,
##not satisfact     embedding = build_embedding()
#    density_estimator_build_fun = posterior_nn(model='nsf',
#                                               hidden_features=128,
#                                               num_transforms=5)
##                                                embedding_net=embedding)
    #density_estimator_build_fun = posterior_nn(model="maf", hidden_features=50)

    density_estimator_build_fun = posterior_nn(model="mdn")


    inference = SNPE(prior=prior, device="cpu",
                     density_estimator=density_estimator_build_fun)


    proposal = prior

    for i in range(num_rounds):

        print(f"Round[{i}]: theta sampling + x simulator")

        if i==0:
            theta, x = simulate_for_sbi(simulator, proposal, num_simulations=num_simu)
        else:
            theta = proposal.sample((num_simu,))
            x = simulate_in_batches(
                simulator, theta, show_progress_bars=True
                )
#            x = simulator(theta)

        print(f"Round[{i}]: density_estimator training")

        density_estimator = inference.append_simulations(
            theta, x, proposal=proposal
            ).train(max_num_epochs=max_num_epochs)
          
        print(f"Round[{i}]: build_posterior")
        posterior = inference.build_posterior(density_estimator)
#                                              sample_with = "mcmc",
#                                              mcmc_method = "slice_np_vectorized",
#                                              mcmc_parameters={"num_chains":100})
        
        print(f"Round[{i}]: new proposal")
        proposal = posterior.set_default_x(x_o)

        with open(output_dir+"/DESXY1_post_SNPE_"+str(i)+"_"+str(num_simu)+"_"+tag+".pkl", "wb") as f:
            pickle.dump(posterior, f)

        cur_state =  get_rng_state()
        with open(output_dir+"/seeds.pkl", 'wb') as f:
            pickle.dump(cur_state, f)


        if do_plot or i==num_rounds-1:
            print(f"Round[{i}]: posterior sampling for plot")
            spls = posterior.sample((10_000,), x=x_o)  # 10_000 independant of num_simu
            spls = spls.cpu()    
            values = [spls[:,i]for i in range(theta_dim)]
            np.save(output_dir+"/DESXY1_values_SNPE_"+str(i)+'_'+str(num_simu)+"_"+tag+'.npy',spls)#

            data = dict(zip(keys,values))
            
            plot_params_kde(data,var_names=keys, figsize=(8,8), limits=None,
                            point_estimate=None, reference_values=truth, reference_color='r',
            patName="SNPE", fname=output_dir+'/fig_DESY1_SNPE_'+str(i)+'_'+str(num_simu)+"_"+tag+'.pdf');



if __name__ == "__main__":

##     run_from_scratch = True

##     if run_from_scratch:
##         print("force seeding from scratch")
##         pyro.set_rng_seed(42)
##     else:
    
##         try:
##             f = open(output_dir+"/seeds.pkl", 'rb')
##             init_state = pickle.load(f)
##             set_rng_state(init_state)
##         except OSError:
##             print("no seeds file, do seeding from scratch")
##             pyro.set_rng_seed(42)

        
    do_multi_pass(num_simu = 100_000, num_rounds = 10, tag="mdn_default",
                  max_num_epochs=None, do_plot=True)
