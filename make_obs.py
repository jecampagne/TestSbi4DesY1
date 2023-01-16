from pathinit import *    # setup  CCIN2P3 env

import os
from astropy.io import fits

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.distributions as torch_dist

from sbi.inference import SNPE, SNLE, SNRE, prepare_for_sbi, simulate_for_sbi
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


####
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("running on device:", device)

####
##  utils
###

from  utils import *


output_dir = "./"


####
# Define an observation
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


z = np.linspace(0,2)
fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
for i in range(4):
    axs[0].plot(nz_source['Z_MID'], nz_source['BIN%d'%(i+1)], color='C%d'%i)
    axs[0].plot(z, nzs_s[i](z), '--', color='C%d'%i)
axs[0].set_xlim(0,2);
axs[0].set_xlabel("z")
axs[0].set_title('Source redshift distributions')

for i in range(5):
    axs[1].plot(nz_lens['Z_MID'], nz_lens['BIN%d'%(i+1)], color='C%d'%i, label=f'FITS BIN[{i}]')
    axs[1].plot(z, nzs_l[i](z), '--', color='C%d'%i, label='jc kde')
    axs[1].set_xlim(0,1.2);
#legend()
axs[1].set_title('Lens redshift distributions');
axs[1].set_xlabel("z");

plt.savefig(output_dir + "/redshift_bins_lens_src.pdf")
plt.close()

# Define some ell range
ell = np.logspace(1, 3)

def model():
    #  Cosmological params
    Omega_c = numpyro.sample('Omega_c', numpyro_dist.Uniform(0.1, 0.9))
    sigma8 = numpyro.sample('sigma8', numpyro_dist.Uniform(0.4, 1.0))
    Omega_b = numpyro.sample('Omega_b', numpyro_dist.Uniform(0.03, 0.07))
    h = numpyro.sample('h', numpyro_dist.Uniform(0.55, 0.91))
    n_s = numpyro.sample('n_s', numpyro_dist.Uniform(0.87, 1.07)) 
    w0 = numpyro.sample('w0', numpyro_dist.Uniform(-2.0, -0.33))

    # Intrinsic Alignment
    A = numpyro.sample('A', numpyro_dist.Uniform(-5., 5.))
    eta = numpyro.sample('eta', numpyro_dist.Uniform(-5., 5.))

    # linear galaxy bias
    bias = [numpyro.sample('b%d'%i, numpyro_dist.Uniform(0.8, 3.0)) 
         for i in range(1,6)]
        
    # parameters for systematics
    m = [numpyro.sample('m%d'%i, numpyro_dist.Normal(0.012, 0.023)) 
         for i in range(1,5)]
    dz1 = numpyro.sample('dz1', numpyro_dist.Normal(0.001, 0.016)) 
    dz2 = numpyro.sample('dz2', numpyro_dist.Normal(-0.019, 0.013)) 
    dz3 = numpyro.sample('dz3', numpyro_dist.Normal(0.009, 0.011)) 
    dz4 = numpyro.sample('dz4', numpyro_dist.Normal(-0.018, 0.022)) 
    dz = [dz1, dz2, dz3, dz4]
    
    # Now that params are defined, here is the forward model
    cosmo = jc.Cosmology(Omega_c=Omega_c, sigma8=sigma8, Omega_b=Omega_b,
                          h=h, n_s=n_s, w0=w0, Omega_k=0., wa=0.)
    
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

    cl, C = jc.angular_cl.gaussian_cl_covariance_and_mean(cosmo, ell, probes, 
                                                          f_sky=0.25, sparse=True)
    P_sparse = jc.sparse.inv(C)
    P = jc.sparse.to_dense(P_sparse)
    C = jc.sparse.to_dense(C)
    return cl, P, C, P_sparse

# So, let's generate the data at the fiducial parameters
fidu_params = {'Omega_c':0.2545, 'sigma8':0.801, 'h':0.682, 'Omega_b':0.0485, 'w0':-1.,'n_s':0.971,
     'A':0.5,'eta':0.,
     'm1':0.0,'m2':0.0,'m3':0.0,'m4':0.0,
     'dz1':0.0,'dz2':0.0,'dz3':0.0,'dz4':0.0,
     'b1':1.2,'b2':1.4,'b3':1.6,'b4':1.8,'b5':2.0
      }
fidu_vals  = [x for x in fidu_params.values()]
fiducial_model = condition(model,fidu_params)

print("Generate one observation... should last 1min or so")
with seed(rng_seed=42):
    mu, P, cov, P_sparse = fiducial_model()
print("done")

# This is our fake data vector
fig = plt.figure()
plt.semilogy(mu);
plt.ylabel("Cls")
plt.savefig(output_dir+"cls.pdf")
plt.close()


#save fidu, P and data
with open('fiducial_dictionary.pkl', 'wb') as f:
    pickle.dump(fidu_params, f)

np.save("precision_mtx.npy",np.asarray(P)) # dense
obs_xo = np.asarray(mu)
np.save("obs_xo.npy",obs_xo)


### Fisher
###

true_theta = np.array([fidu_params["Omega_c"],fidu_params["sigma8"], fidu_params["w0"], fidu_params["h"],
                       fidu_params["Omega_b"], fidu_params["n_s"],
                       fidu_params["A"], fidu_params["eta"],
                       fidu_params["b1"], fidu_params["b2"], fidu_params["b3"], fidu_params["b4"],fidu_params["b5"],
                       fidu_params["m1"], fidu_params["m2"], fidu_params["m3"], fidu_params["m4"],
                       fidu_params["dz1"], fidu_params["dz2"], fidu_params["dz3"],  fidu_params["dz4"]])


def unpack_params_vec(theta):

  # Retrieve cosmology
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


# We define a parameter dependent function that computes the mean (aka cl)
@jax.jit
def jc_mean(p):

    cosmo, m, dz, [A, eta], bias = unpack_params_vec(p)

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

    # Compute signal vector
    mu = jc.angular_cl.angular_cl(cosmo, ell, probes)

    
    # We want mu in 1d to operate against the covariance matrix
    return mu.flatten() 


jc_mu = jc_mean(true_theta)

# in principle we should get the same as above
print("Verif; ", np.allclose(np.asarray(jc_mu),obs_xo))

# We compute it's jacobian with JAX, and we JIT it for efficiency
jac_mean = jax.jit(jax.jacfwd(jc_mean))
 
# We can now evaluate the jacobian at the fiducial cosmology
dmu = jac_mean(true_theta)

print("dmu shape:",dmu.shape)


# Fisher Matrix is then
print("Fisher mtx comp...")
F_mtx = jc.sparse.dot(dmu.T, P_sparse, dmu)

print("Fisher shape:",F_mtx.shape)
np.save("fisher_mtx.npy",np.asarray(F_mtx))

## MOPED


Nparams = true_theta.shape[0]
Ndata   = mu.shape[0]

B_mtx = jnp.zeros(shape=(Nparams,Ndata))

def body(i,val):
    #decode val
    B_mtx, P_sparse, dmu, F_mtx = val
    
    scal_prod = B_mtx @ dmu[:,i]
    tmp = (scal_prod[jnp.newaxis] @ B_mtx).squeeze()   #or use jnp.einsum('i,ij',scal_prod,B_mtx)
    
    B_mtx = B_mtx.at[i,:].set(
        (jc.sparse.dot(P_sparse,dmu[:,i]) - tmp)/jnp.sqrt(F_mtx[i,i]- scal_prod @ scal_prod)
        )

    #return val
    return  B_mtx, P_sparse, dmu, F_mtx

val_init = (B_mtx, P_sparse, dmu, F_mtx)
val = jax.lax.fori_loop(0,Nparams,body, val_init)
B_mtx, P_sparse, dmu, F_mtx = val

#print("B_mtx: ",B_mtx)
#print(B_mtx @ cov @ B_mtx.T)
print("Verif B_mtx @ cov @ B_mtx.T = Id", jnp.allclose(B_mtx @ cov @ B_mtx.T, jnp.eye(Nparams) ))


mu_compressed = B_mtx @ mu
np.save("obs_xo_comp.npy",np.asarray(mu_compressed))
np.save("moped_vect.npy",np.asarray(B_mtx))
