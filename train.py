# -*- coding: utf-8 -*-

# experiments were run with Python 3.6, PyTorch 1.1.0, Linux CUDA 9.0


### data was obtained from : Mottini, Alejandro, and Rodrigo Acuna-Agost. "Deep choice model using pointer networks for airline itinerary prediction." Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2017. 
### results of the paper were obtained with:
### python train.py  --batch_size  16 --activation 3  --lr .001 --hidden_layers 2 --nodes_per_layer 512 --final  --max_epochs 66  
### python eval_test.py --batch_size 16 --model models/pcmcNet_device~cuda_patience~5_sig_imp~0.01_dev_batch_size~8_activation~3_train_batch_size~16_max_epochs~66_hidden_layers~2_lr~0.001_index~0_nodes_per_layer~512_dropout~0.5.pth 


import torch
import torch.nn as nn

import time

### exact reproducibility is not guaranteed by PyTorch : https://pytorch.org/docs/stable/notes/randomness.html
from numpy.random import seed
seed(12345)
torch.manual_seed(12345)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from argparse import ArgumentParser

from torch_modules import TrainableLayers, ChoiceDataset, my_collate, pcmc_net_batch

from torch.utils.data import DataLoader


import sys
import pickle
import os

class Trainer():
    def __init__(self,final, num_workers):
        self.context_cat_features = ["OD","officeID"]
        self.context_num_features = ['staySaturday','isContinental','isDomestic','depWeekDay','dtd']
        
        
        self.alt_num_features = [ 'stayDurationMinutes', 'totalPrice','totalTripDurationMinutes', 'nAirlines', 'nFlights', 'outDepTime', 'outArrTime']
        
        self.alt_cat_features = ["firstAirline"]
        
        self.target = "choice"
        
        
        self.trainset = None
        self.devset = None
        self.cat_cardinality = None
        
        self.final = final
        self.num_workers = num_workers

    def load_data(self,dir_name):
        
        train = pd.read_csv(os.path.join(dir_name, "data_train_50alternatives-EUR-TopOD_filter.csv"))
        dev = pd.read_csv(os.path.join(dir_name, "data_dev_50alternatives-EUR-TopOD_filter.csv"))
        test = pd.read_csv(os.path.join(dir_name, "data_test_50alternatives-EUR-TopOD_filter.csv"))

        
        train.set_index(["individual","alternative"], inplace=True)
        dev.set_index(["individual","alternative"], inplace=True)
        test.set_index(["individual","alternative"], inplace=True)
        
        learnset = pd.concat([train,dev,test], axis=0) 

        learnset["firstAirline"] = learnset["airlines"].str.split("/").apply(lambda x:x[0])

        preproc = {}    
        
        ## now preprocess the other features
        
        for c in self.context_cat_features + self.alt_cat_features:
            e = LabelEncoder()
            learnset[c] = e.fit_transform(learnset[c]) #categorical features must be longInt for embedding layer
            preproc[c] = e
        
        
        
        learnset = learnset[self.context_cat_features+self.alt_cat_features+self.context_num_features+self.alt_num_features+  ["choice"]]
        
        
        self.cat_cardinality = learnset[self.context_cat_features+self.alt_cat_features].nunique()
    
        
        if len(self.context_num_features+self.alt_num_features)>0:
            scl = StandardScaler()
            learnset[self.context_num_features+self.alt_num_features] = scl.fit_transform(learnset[self.context_num_features+self.alt_num_features])
            preproc["num_features"]=scl


        if self.final:
            testset = learnset[learnset.index.isin(test.index)]
            print("testset.shape: ",testset.shape)
            pickle.dump( testset, open( "testset.p", "wb" ) )
            print("testset dumped")
            
            pickle.dump( preproc, open( "preprocessors.p", "wb" ) )
            print("preprocessors saved")
            
        
        learnset.reset_index(inplace=True) #put individual.alternative as column
        
        trainset = learnset[:train.shape[0]]
        devset = learnset[train.shape[0]:]
    
        trainset = [df for _,df in trainset.groupby("individual")]
        devset = [df for _,df in devset.groupby("individual")]
    
    
        if self.final:
            trainset+=devset
        else:
            print("devset len: ",len(devset))
            self.devset = ChoiceDataset(devset, ctxt_cat_cols=self.context_cat_features, ctxt_num_cols=self.context_num_features,
                           alt_cat_cols=self.alt_cat_features, alt_num_cols=self.alt_num_features)

        print("trainset len: ",len(trainset))
    
        self.trainset = ChoiceDataset(trainset, ctxt_cat_cols=self.context_cat_features,
                                      ctxt_num_cols=self.context_num_features, alt_cat_cols=self.alt_cat_features, 
                                      alt_num_cols=self.alt_num_features)
    


    def emb_tuple(self, card):
        return (card,int(min(np.ceil((card)/2), 50 )))

        
        
    def run(self,index, lr, dropout, hidden_layers, 
            nodes_per_layer, max_epochs, train_batch_size, activation, dev_batch_size,
            sig_imp=None,patience=None): 


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(device)

        
        #get all parameters except self
        loc = locals().copy()  
        del loc["self"]
        
        str_model = "pcmcNet_" + "_".join([k + "~" + str(v) for  k,v in loc.items()])
        
        print("building model: ", str_model)
        
        sys.stdout = open(str_model + ".out", "w")

        dropout = torch.tensor(dropout).to(device)
        
        hiddenLayersDim = [nodes_per_layer for e in range(hidden_layers)]
        lin_layer_dropouts = [dropout for e in range(hidden_layers)]
        emb_dropout = dropout
        
        
        ctxt_emb_dims = [self.emb_tuple(self.cat_cardinality[c])  for c in self.context_cat_features]
        ctxt_no_of_cont = len(self.context_num_features)
        alt_emb_dims = [self.emb_tuple(self.cat_cardinality[c])  for c in self.alt_cat_features]
        alt_no_of_cont = len(self.alt_num_features)
        

        pickle.dump( (hiddenLayersDim,lin_layer_dropouts,emb_dropout,ctxt_emb_dims,ctxt_no_of_cont,alt_emb_dims,alt_no_of_cont,activation),
               open( "architecture_params.p", "wb" ) )
        
        
        model = TrainableLayers(ctxt_emb_dims=ctxt_emb_dims, ctxt_no_of_cont=ctxt_no_of_cont, 
                           alt_emb_dims=alt_emb_dims, alt_no_of_cont=alt_no_of_cont, lin_layer_sizes=hiddenLayersDim,
                          emb_dropout=emb_dropout, lin_layer_dropouts=lin_layer_dropouts,activation=activation).to(device)
        print(model)
        
        
        
        
        #######
        
        num_workers = self.num_workers
        
        
    
        ### IMPORTANT: batch_size is in number of whole sessions
        train_loader = DataLoader(self.trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=my_collate) 
        
        loss = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        
        if not self.final: 
            dev_loader = DataLoader(self.devset, batch_size=dev_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=my_collate) 

            
        
        
        ############# MAIN LOOP #############
        
        since = time.time()
        
        train_losses, dev_losses = [], []
        print("starting training" )
        patience_count = 0
        for epoch in range(max_epochs):

            if not self.final:
                if len(dev_losses)>1:
                    if(dev_losses[-1]>min(dev_losses[:-1])-sig_imp):
                        patience_count+=1
                        if patience_count==patience:
                            print("EARLY STOPPING")
                            break
                    else:
                        patience_count=0            

            
            for i, (ys,xs,sessions_sizes) in enumerate(train_loader):

                batchloss = pcmc_net_batch(model,device,ys,xs,sessions_sizes)[0]
                
                # Backward Pass and Optimization
                optimizer.zero_grad()
                batchloss.backward()
                optimizer.step()
                
            
            # Turn off gradients
            with torch.no_grad():
                # set model to evaluation mode: no dropout
                model.eval()
                
                training_loss = 0.0
                for i, (ys,xs,sessions_sizes) in enumerate(train_loader):
                   
                    batchloss = pcmc_net_batch(model,device,ys,xs,sessions_sizes)[0]
                    training_loss += batchloss.item()
                   
                train_losses.append(training_loss/len(train_loader))                

                if not self.final:
                    dev_loss = 0.0
                    for i, (ys,xs,sessions_sizes) in enumerate(dev_loader):
                    
                       batchloss = pcmc_net_batch(model,device,ys,xs,sessions_sizes)[0]
                    
                       dev_loss += batchloss.item()
                    
                    dev_losses.append(dev_loss/len(dev_loader)) 

                
                # set model back to train mode
                model.train()

                print("")
                
                if not self.final:
                    print("Epoch: {}/{}".format(epoch + 1, max_epochs),
                      "Training loss: {:.2f}".format(train_losses[-1]),
                      "Dev Loss: {:.2f}".format(dev_losses[-1]))
                else:
                    print("Epoch: {}/{}".format(epoch + 1, max_epochs),
                      "Training loss: {:.2f}".format(train_losses[-1]))
                    
                time_elapsed = time.time() - since
                print('Time since beginning {:.0f}m {:.0f}s'.format(
                        time_elapsed // 60, time_elapsed % 60))
                sys.stdout.flush()
             
                   
        
        print('Finished Training')
        

        ############# END MAIN LOOP #############
    
    
        sys.stdout = sys.__stdout__
        
        if self.final:
            directory = './models/' # +str(index) + "/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(model.state_dict(), directory + str_model+".pth")
            return None
        else:
            return dev_losses[-1]
        
        





if __name__ == "__main__":
    
    def to_enum(l):
        return tuple(range(l[0],l[1]+1))
    
    
    parser = ArgumentParser()

    parser.add_argument('--data_folder', type=str, default="~/data/choice/",
                        help='folder containing data (default: %(default)s)')


    parser.add_argument("--optimize", help="hypeparameter bayesian optimization",
                    action="store_true")

    parser.add_argument("--final", help="merge train and dev for single run",
                    action="store_true")

    
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: %(default)s)')
    parser.add_argument('--dev_batch_size', type=int, default=8,
                        help='input batch size for validation (default: %(default)s)')
    parser.add_argument('--max_epochs', type=int, default=1000,
                        help='max number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=.01,
                        help='learning rate (default: %(default)s)')
    parser.add_argument('--dropout', type=float, default=.5,
                        help='dropout (default: %(default)s)')
    parser.add_argument('--hidden_layers', type=int, default=1,
                        help='number of hidden layers (default: %(default)s)')
    parser.add_argument('--nodes_per_layer', type=int, default=32,
                        help='number of nodes per hidden layer (default: %(default)s)')

    parser.add_argument('--activation', type=int, default=0,
                        help='activation: 0=relu, 1=sigmoid, 2=tanh, 3=leakyrelu (default: %(default)s)')


    parser.add_argument('--num_cores', type=int, default=7,
                        help='number of cores for GPyOpt (default: %(default)s)')
    parser.add_argument('--num_workers', type=int, default=7,
                        help='number of workers (default: %(default)s)')

    parser.add_argument('--sig_imp', type=float, default=.01,
                        help='minimum improvement considered as significant for early stopping (default: %(default)s)')

    parser.add_argument('--patience', type=int, default=5,
                        help='patience for early stopping (default: %(default)s)')


    parser.add_argument ('--opt_lr_np10',  nargs=2, type=int)
    parser.add_argument ('--opt_hid_lay',  nargs=2, type=int)
    parser.add_argument ('--opt_nodes_p2', nargs=2, type=int)
    parser.add_argument ('--opt_bsize_p2', nargs=2, type=int)

    parser.add_argument('--opt_max_iter', type=int, default=20,
                        help='max number of iterations for optimization (default: %(default)s)')

    args = parser.parse_args()

    if args.final and args.optimize:
        exit("cant optimize and be final")


    t = Trainer(args.final, args.num_workers)
    t.load_data(args.data_folder)


    if(args.optimize):
        
        from GPyOpt.methods import BayesianOptimization
        
        
        bds = [{'name': 'lr_neg_pow_10', 'type': 'discrete', 'domain': to_enum(args.opt_lr_np10)},
               {'name': 'hidden_layers', 'type': 'discrete', 'domain': to_enum(args.opt_hid_lay)},
               {'name': 'nodes_per_layer_power_2', 'type': 'discrete', 'domain': to_enum(args.opt_nodes_p2)},
               {'name': 'batch_size_pow_2', 'type': 'discrete', 'domain': to_enum(args.opt_bsize_p2)},
               {'name': 'activation', 'type': 'discrete', 'domain': to_enum((0,3))}]
        

        print(bds)
        print("max_iter: ",args.opt_max_iter )
        sys.stdout.flush()
        
        def fun(params):
            fun.index += 1
            params = params[0]
            print("evaluating config: ",params)

            res =  t.run(index=fun.index, lr=10**-float(params[0]), dropout=.5, hidden_layers=int(params[1]),nodes_per_layer=int(2**params[2]), max_epochs=args.max_epochs,
                         train_batch_size=2**int(params[3]), activation=int(params[4]), dev_batch_size=args.dev_batch_size,sig_imp=args.sig_imp,patience=args.patience)
            print("dev loss: ",res)
            return res
        fun.index = 0
        
        
        optimizer = BayesianOptimization(f=fun, 
                                         domain=bds,
                                         model_type='GP',
                                         acquisition_type ='EI',
                                         acquisition_jitter = 0.05,
                                         exact_feval=True, 
                                         maximize=False,
                                         verbosity=True,
                                         de_duplication=True) 
        
        # we have 5 initial random points
        optimizer.run_optimization(max_iter=args.opt_max_iter, verbosity=True, report_file="optim_results.txt")
        
        print("best params: ", optimizer.x_opt)
        print("best dev loss: ", optimizer.fx_opt)

    else:
        t.run(0, args.lr,  args.dropout, args.hidden_layers, args.nodes_per_layer, max_epochs=args.max_epochs,
              train_batch_size=args.batch_size, activation=args.activation, dev_batch_size=args.dev_batch_size,sig_imp=args.sig_imp,patience=args.patience)




