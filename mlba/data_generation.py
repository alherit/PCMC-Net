import numpy as np
from pymlba import predict
import pandas as pd

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--seed', type=int, default=1234,
                    help='seed for data generation (default: %(default)s)')

parser.add_argument('-N', type=int, default=20000,
                    help='sample size to be generated (default: %(default)s)')



args = parser.parse_args()

N = args.N
seed = args.seed

np.random.seed(seed)


D = np.random.uniform(1.,9., size= N*2).reshape(-1,2)
X = np.tile([4.,6.],[D.shape[0],1])
Y = np.tile([6.,4.],[D.shape[0],1])
M = np.concatenate([X,Y,D], axis=1).reshape(-1,3,2)

probs = predict(M, 5, .2, .4, 5.)

I = np.eye(3, dtype=int)
choice =  [I[np.random.choice(3, size=1, p=p / p.sum())] for p in probs] ##normalization is needed to fix precision issues
choice = np.concatenate(choice).reshape(-1,1)


probs = np.stack(probs, axis=0)


indiv = np.expand_dims(np.repeat(np.arange(N),3), axis=1)
alt = np.expand_dims(np.tile(np.arange(3),N), axis=1)
probs = np.expand_dims(np.concatenate(probs), axis=1)

df = pd.DataFrame(np.concatenate([np.concatenate(M), probs], axis=1), columns=["attr1", "attr2", "true_prob"])

df["indiv"] = indiv
df["alt"] = alt
df["choice"] = choice

fname = "generated_N"+str(N)+"_seed"+str(seed)+".bz2"

df.to_pickle(fname, protocol=2)

fname_csv = "generated_N"+str(N)+"_seed"+str(seed)+".csv"


df.to_csv(fname_csv, index=False)





