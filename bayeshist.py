#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
import sys
import time

# 	Input 1 (required) : file name for inputs (likelihoods)
# 	Input 2 (required) : number of Gibbs samples 
# 	Input 3 (optional) : file name for input initialisation

# Parse inputs
assert len(sys.argv) <= 3, "The number of inputs should be <= 3"
infilename = sys.argv[1]
nsamples = int(sys.argv[2])
if len(sys.argv) > 3:
    fname_ini = sys.argv[3]
burnin_fraction = 0.3

fname = infilename + '_post.npy'

comm = MPI.COMM_WORLD
MPI_size = comm.Get_size()
MPI_rank = comm.Get_rank()

if MPI_rank == 0:
    print('Input parameters:', sys.argv)
    print('Running on %d cores' % MPI_size)
    
# This is the dirichlet prior. rsize is the size of the output per np.random.gamma call. It is set to 1 in all cases used here.
# It returns a set of Dirichlet-distributed numbers. According to wikipedia you can sample vectors of Dirichlet distributed 
# numbers by using the Gamma distribution https://en.wikipedia.org/wiki/Dirichlet_distribution#Random_number_generation

def dirichlet(rsize, alphas):
    gammabs = np.array([np.random.gamma(alpha+1, size=rsize) for alpha in alphas])
    fbs = gammabs / gammabs.sum(axis=0)
    return fbs.T

# Load the chuncks created in the jupyter notebook based on the MPI rank. Each rank reads a different chunk of the data.

pdfints_npfile = infilename+'_'+str(MPI_rank+1)+'.npy'
pdfints = np.load(pdfints_npfile)
sh = pdfints.shape
nobj = sh[0]
print('Read file', pdfints_npfile, 'and found', nobj, 'objects')
nbins = np.prod(sh[1:])
pdfints = pdfints.reshape((nobj, nbins))

# Depending on the inputs given to the script, we either randomly initialize the proposed values for the {f_ijk} or we read
# them from an input file.

if MPI_rank == 0:
    if len(sys.argv) > 3:
        print('Initialised sampler with file', fname_ini)
        hbs = np.load(fname_ini).reshape((nbins,))
        hbs /= hbs.sum()
    else:# random initialisation
        nbs = np.random.rand(nbins)
        hbs = dirichlet(1, nbs)
else:
    hbs = None

# Wait for all processes to read their respective chunks.
comm.Barrier()

# Broadcast the initialized hbs (Hierarchical bayesian sample?) to all processes, so all are working with the same values
hbs = comm.bcast(hbs, root=0)


if MPI_rank == 0:
    print('Broadcasted hbs')

# MPI_rank 0 is in charge of writing to the buffer
if MPI_rank == 0:
    fbs = np.zeros( (nsamples, nbins) )
    tstart = time.time()

    
# Wait for all processes to read hbs and for rank 0 to finish printing and creating the buffer
comm.Barrier()

# This is a helper array that generates a large matrix with the number of the indexes of the bin.
# It has dimensions nobj * nbins    ...  (nobj * (nz*nt*nm))
ibins = np.repeat(np.arange(1, nbins), nobj).reshape((nbins-1, nobj)).T.ravel()

# We will generate nsample samples, but a certain fraction will be used for burn in. Which can be changed in the configuration
# above.
for kk in range(1, nsamples):

    
    #### This is the key part. Here is where the likelihood function enters. Here we multiply the sampled f_ijk by the
    #### likelihood. The input files are not the data itself, but the likelhood generated FROM the data.
    prods = (pdfints * hbs) # Eq (12) of  Leistedt, Mortlock & Peiris. 
    
    
    # For each object, flatten the previous PDF and multply for the sampled HBS. This yields a 1D array of length 'nbins' with
    # probabilities for each object.
    # For each flattened pdf, generate a random number from U(0,1)
    # Find the index in the flattened array at which the cumulative probability surpasses the random number. Then add 1 to
    # that bin in the n_ijk.
    
    
    cumsumweights = np.add.accumulate(prods, axis=1).T #cumsumweights = prods.cumsum(axis=1).T 
    cumsumweights /= cumsumweights[-1,:]
    pos = np.random.uniform(0.0, 1.0, size=nobj) # random uniformly sampled value for each object in the sample
    cond = np.logical_and(pos > cumsumweights[:-1,:], pos <= cumsumweights[1:,:])
    res = np.zeros(nobj, dtype=int)
    res[pos <= cumsumweights[0,:]] = 0
    locs = np.any(cond, axis=0)
    res[locs] = ibins[cond.T.ravel()]
    ind_inrange = np.logical_and(res > 0, res < nbins)
    nbs = np.bincount(res[ind_inrange], minlength=nbins) # Here nbs is n_ijk

    nbs_all = np.zeros_like(nbs)
    # We sum over all processes and get the value
    comm.Allreduce(nbs, nbs_all, op=MPI.SUM) 
    
    if MPI_rank == 0:

        if kk % 100 == 0:
            #print kk
            tend = time.time()
#             fname = infilename + '_post.npy'
            ss = int(burnin_fraction*kk)
            sh2 = tuple([kk-ss]+list(sh[1:]))
            print('Saving', kk-ss, 'samples to', fname, '(%.2f' % (float(tend-tstart)/kk), 'sec per sample)')
#             np.save(fname, fbs[ss:kk, :].reshape(sh2))

        # Since we already have casted the value of n_ijk by summing over all processes, we obtain the new random dirichlet
        # Sample. Question: Why is it made using n_ijk rather than n_ijk/Nobj? Or it doesn't make a difference?
        hbs = dirichlet(1, nbs_all) #### PLUS ONE HERE OR NOT?? ???? ??? 

    hbs = comm.bcast(hbs, root=0)

    if MPI_rank == 0:
        fbs[kk,:] = hbs
        
comm.Barrier()
if MPI_rank == 0:        
    np.save(fname, fbs[:, :].reshape([nsamples]+list(sh[1:])))