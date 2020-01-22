# (c) 2020 alherit
# This code is licensed under MIT license (see LICENSE.txt for details)

import larch
import pandas as pd
import numpy as np
from larch.roles import PX

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--trainset', type=str, default=None,
                    help='testset `csv file (default: %(default)s)')

parser.add_argument('--testset', type=str, default=None,
                    help='testset pickle file (default: %(default)s)')


args = parser.parse_args()

def aic(loglike,nparams):
    return 2*nparams-2*loglike

def fitModel(d,vars):
    m = larch.Model(d)
    m.option.threads=2
    m.utility.ca = sum(PX(i) for i in vars)
    result = m.maximize_loglike()
    
    print(m.report('txt', sigfigs=3))
    
    print(vars)
    print("loglike: ",result.loglike)
    nparams = len(vars)
    print("nparams: ",nparams)
    print("aic: ",aic(result.loglike,nparams))
    return m



indiv_col = 'indiv'
alt_col = 'alt'
choice_col = 'choice'

d = larch.DB.CSV_idca(args.trainset,caseid=indiv_col, altid=alt_col, choice=choice_col, tablename="train")



#%%


vars = ['attr1','attr2']

m = fitModel(d,vars)

#%%

data_test = pd.read_pickle(args.testset)

#%%
coef=[m[var].value for var in vars]


def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def applyCoefs(row,variables,coef):
    #print(row[['individual','choice']])
    row_values=row[variables].values
    #print(row_values)
    logit=np.sum(np.multiply(row_values,coef) )
    row['choice_prob']=logit
    #print(logit)
    return row


data_test['choice_prob']=0


unique_ind_test=data_test[indiv_col].unique()
for indiv in unique_ind_test:
    data_indiv=data_test[data_test[indiv_col]==indiv].copy(deep=True)
    data_indiv.update(data_indiv.apply(lambda row:applyCoefs(row,variables=vars,coef=coef) ,axis=1))#per row
    
    data_indiv['choice_prob']= softmax(data_indiv['choice_prob'].values)
 
    data_test.loc[data_test[indiv_col]==indiv,'choice_prob']=data_indiv['choice_prob']
    
    
#%%

probs = data_test.loc[data_test["choice"]==1,"choice_prob" ]
print(-np.log(probs).mean())




#%%

import scipy.stats


def conditionalKL(df,probcol):
    return df.groupby('indiv').apply(lambda x: scipy.stats.entropy(x["true_prob"].values,x[probcol].values)).mean()

print("KL: ", conditionalKL(data_test,"choice_prob"))


#%%

import numpy as np


#Figure 7
X = [4.,6.]
Y = [6.,4.]


prefX = np.exp(np.dot(coef,X)) / (np.exp(np.dot(coef,X)) + np.exp(np.dot(coef,Y)))


#%% 


D = np.mgrid[1:9.1:.1, 1:9.1:.1].reshape(2, -1).T

Dlen = int(round(np.sqrt(D.shape[0])))

    

from pylab import imsave

from pylab import imshow, text, plot, savefig, scatter
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.clf()

im = imshow(np.tile(prefX, [Dlen,Dlen]),cmap='gray', vmin=0., vmax=1., origin="lower", extent=[1,9,1,9])
savefig("mnl_N20000_decorated.pdf",bbox_inches='tight')

im = imsave("mnl_N20000.pdf",np.tile(prefX, [Dlen,Dlen]),cmap='gray', vmin=0., vmax=1., origin="lower")




    