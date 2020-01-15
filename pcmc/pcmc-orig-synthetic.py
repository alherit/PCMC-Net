# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:08:51 2019

@author: alheritier
"""

# python 2
# requires pcmc-nips\\lib from https://github.com/sragain/pcmc-nips in PYTHONPATH

import pickle


import pandas as pd
import numpy as np
from pcmc_utils import infer,solve_ctmc, comp_Q

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--trainset', type=str, default=None,
                    help='testset pickle file (default: %(default)s)')

parser.add_argument('--testset', type=str, default=None,
                    help='testset pickle file (default: %(default)s)')


args = parser.parse_args()



fname = args.trainset
print(fname)
trainset = pd.read_pickle(fname)




# training data is a dictionary of counts for each choice set made of identified alternatives

def id1(nbins):
    return nbins*nbins

def id2(nbins):
    return nbins*nbins+1


def encode(x,nbins):
    bins = np.linspace(1, 9, num=nbins+1)
    return nbins*(np.digitize(x['attr1'].iloc[2], bins)-1) +  np.digitize(x['attr2'].iloc[2], bins)-1

def build_Cset(df,alpha,nbins):
    train_choices = trainset.groupby('indiv').apply(lambda x: x['choice'].values)
    
    train_keys = trainset.groupby('indiv').apply(lambda x: encode(x,nbins))
    
    df = pd.DataFrame()
    
    df['id3'] = train_keys
    df['id1'] = id1(nbins)
    df['id2'] = id2(nbins)
    
    df['choice'] =  train_choices
    
    grouped = df.groupby(['id1','id2','id3']).apply(lambda x: pd.Series({'counts': np.sum(x['choice'].values)+alpha}))
    
    C = grouped['counts'].to_dict()
    
    return C

#regularization: additive smoothing
alpha=.1 
nbins = 8

Ctrain = build_Cset(trainset,alpha,nbins)

#x: starting parameters
#maxiter- number of iterations allowed to optimizer
maxiter = 25 # default value in pcmc-nips code
#n- number of elements in universe
# discretize third option


pcmc_params = infer(C=Ctrain,x=None,n=nbins*nbins+2,maxiter=maxiter,delta=1)

pickle.dump(pcmc_params,open("pcmc_N10000_params_nbins"+str(nbins)+".p", "wb"))

pcmc_params =pickle.load(open("pcmc_N10000_params_nbins"+str(nbins)+".p", "rb"))

fname = args.testset
print(fname)
testset = pd.read_pickle(fname)


Q = comp_Q(pcmc_params)



def getProb(group,nbins):
    x = (id1(nbins), id2(nbins), encode(group, nbins))
    return solve_ctmc(Q[x,:][:,x])    

import scipy.stats

print "kl div nbins "+str(nbins)+" wrt True model" , testset.groupby('indiv').apply(lambda x: scipy.stats.entropy(x["true_prob"].values,getProb(x,nbins))).mean()

def encode2(x,nbins):
    bins = np.linspace(1, 9, num=nbins+1)
    return nbins*(np.digitize(x[0], bins)-1) +  np.digitize(x[1], bins)-1


def getProb2(third, nbins):
    x = (id1(nbins), id2(nbins), encode2(third, nbins))
    return solve_ctmc(Q[x,:][:,x])    


#plot empirical distribution discretized
d = { (i/nbins,i%nbins):v for (_,_,i),v in Ctrain.items()}
prob_emp = np.stack([d[k] for k in sorted(d.keys())])
prefX = prob_emp [:,0] /(prob_emp [:,0] + prob_emp [:,1])

from pylab import imsave

Z = prefX.reshape(int(round(np.sqrt(len(prefX)))),-1).T

im = imsave("empirical_N20000_nbins"+str(nbins)+".pdf",Z,cmap='gray', vmin=0., vmax=1., origin="lower")


#Figure 2
D = np.mgrid[1:9:.1, 1:9:.1].reshape(2, -1).T



probs = np.apply_along_axis(lambda r:getProb2(r,nbins), 1, D)


prefX = probs[:,0] /(probs[:,0] + probs[:,1])


from pylab import imsave, imshow, savefig

Z = prefX.reshape(int(round(np.sqrt(len(prefX)))),-1).T

im = imsave("pcmc-orig_N20000_nbins"+str(nbins)+".pdf",Z,cmap='gray', vmin=0., vmax=1., origin="lower")

import matplotlib.pyplot as plt
plt.clf()

plt.xticks(np.arange(1,10,1))
im = imshow(Z,cmap='gray', vmin=0., vmax=1., origin="lower", extent=[1,9,1,9])
savefig("pcmc-orig_N20000_nbins"+str(nbins)+"_decorated.pdf",bbox_inches='tight')


