# (c) 2020 alherit
# This code is licensed under MIT license (see LICENSE.txt for details)

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

import os

import torch

from numpy.random import seed
seed(12345)
torch.manual_seed(12345)


from torch.utils.data import DataLoader
from torch_modules import TrainableLayers, ChoiceDataset,my_collate,pcmc_net_batch

from argparse import ArgumentParser

context_cat_features = []
context_num_features = []
      
alt_num_features = ['attr1','attr2']
alt_cat_features = []


parser = ArgumentParser()


parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for testing (default: %(default)s)')



parser.add_argument('--arch', type=str, default="architecture_params.p",
                    help='model architecture params pickle file (default: %(default)s)')

parser.add_argument('--model', type=str, default=None,
                    help='trained model pth file (default: %(default)s)')

parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers (default: %(default)s)')



args = parser.parse_args()

(hiddenLayersDim,lin_layer_dropouts,emb_dropout,ctxt_emb_dims,
 ctxt_no_of_cont,alt_emb_dims,alt_no_of_cont,activation) = pickle.load( open( args.arch, "rb" ))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device: " , device)

# dropout needs to be moved to the current device, because BinomialDistribution uses this information to generate
# the random tensors 
lin_layer_dropouts = [dropout.to(device) for dropout in lin_layer_dropouts]
emb_dropout = emb_dropout.to(device)


torch.set_num_threads(args.num_workers)

model = TrainableLayers(ctxt_emb_dims=ctxt_emb_dims, ctxt_no_of_cont=ctxt_no_of_cont, 
                           alt_emb_dims=alt_emb_dims, alt_no_of_cont=alt_no_of_cont, lin_layer_sizes=hiddenLayersDim,
                       emb_dropout=emb_dropout, lin_layer_dropouts=lin_layer_dropouts,activation=activation).to(device)


model.load_state_dict(torch.load(args.model,map_location=device))



D = np.mgrid[1:9.1:.1, 1:9.1:.1].reshape(2, -1).T



#Figure 7
X = np.tile([4.,6.],[D.shape[0],1])
Y = np.tile([6.,4.],[D.shape[0],1])




M = np.concatenate([X,Y,D], axis=1).reshape(-1,2)


testdf = pd.DataFrame(M, columns=["attr1","attr2"])

#normalize
testdf[alt_num_features] = (testdf[alt_num_features] - 1.)/8. 


testdf["alt"] = np.tile([0,1,2],D.shape[0])

testdf["individual"] = np.repeat(np.arange(D.shape[0]),3)

#  fake: just for the shape and to have 1 choice only, otherwise it breaks
testdf["choice" ]  = np.tile([0,1,2],D.shape[0])


testset = [df for _,df in testdf.groupby("individual")]
## groupby sorts key by default, so we need to sort the dataframe to put the probs in the correct lines
testdf =  testdf.sort_index(level="individual")

testset = ChoiceDataset(testset, ctxt_cat_cols=context_cat_features, ctxt_num_cols=context_num_features,
                           alt_cat_cols=alt_cat_features, alt_num_cols=alt_num_features)


test_loader = DataLoader(testset, args.batch_size, shuffle=False, pin_memory=True, collate_fn=my_collate, num_workers=args.num_workers) 

all_preds = []

# Turn off gradients
with torch.no_grad():
    # set model to evaluation mode: no dropout
    model.eval()
    
    test_loss = 0.0
    for i, (ys,xs,sessions_sizes) in tqdm(enumerate(test_loader)):
       
        _, _, batchlogprobs = pcmc_net_batch(model,device,ys,xs,sessions_sizes)
        all_preds += batchlogprobs 
       

testdf["prob"] = np.exp(torch.cat(all_preds).cpu().detach().numpy())



from pylab import imsave, imshow, savefig

import matplotlib
matplotlib.use('Agg')


import os
modelstr = os.path.basename(args.model)

prefX = testdf.groupby("individual").apply(lambda x: x["prob"].values[0]/(x["prob"].values[0] + x["prob"].values[1]) ).values

Z = prefX.reshape(int(round(np.sqrt(len(prefX)))),-1).T

im = imsave("pcmcnet"+modelstr+".pdf",Z,cmap='gray', vmin=0., vmax=1., origin="lower")


im = imshow(Z, cmap='gray', vmin=0., vmax=1., origin="lower", extent=[1,9,1,9])
savefig("pcmcnet"+modelstr+"_decorated.pdf",bbox_inches='tight')
