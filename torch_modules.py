# (c) 2020 alherit
# This code is licensed under MIT license (see LICENSE.txt for details)

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.utils.data import Dataset, DataLoader


def pcmc_net_batch(model,device,ys,xs,sessions_sizes):
        """
        Do a forward pass on pcmc-net using the given model containing the representation, cartesian product and transition rate layers

        Parameters
        ----------

        model:  nn.Module
          contains representation, cartesian product and transition layers

        device: device where pytorch is running

        ys: 2D array of int
          contain the chosen index for each session in the batch

        xs: tuple of numeric 2D arrays (int for categorical, float for numerical)
          each array correspond to each category of feature (ctxt_cont_data, ctxt_cat_data, alt_cont_data, alt_cat_data)
          context arrays have one row per session of the batch

        sessions_sizes: list of int
          contains the number of alternatives for each session in the batch

        """
        xs = tuple(item.to(device, non_blocking=True) for item in xs)
        ys = ys.to(device, non_blocking=True)

        ys = torch.split(ys.reshape(-1),1)

        q_ij = model(xs,sessions_sizes,device)[:,0]

        sq_sessions_sizes = [s*s for s in sessions_sizes]
        split_q_ij = torch.split(q_ij,sq_sessions_sizes)

        # elements in q_ij are supposed to be in the order 11,12,23.. 21,22,23 ... 
        #reshape results into matrix and remove diagonal
        Qs = [t.reshape(s,s) * (1-torch.eye(s).to(device)) for t,s in zip(split_q_ij,sessions_sizes)]
        #diagonal must be minus sum of row
        Qs = [q - torch.eye(q.shape[0]).to(device)*torch.sum(q,dim=1) for q in Qs] 
        #replace last column with ones    
        Qs = [torch.cat([q[:,:-1] , torch.ones(q.shape[0]).to(device).unsqueeze(1)], dim=1) for q in Qs]

        Bs= [torch.cat([torch.zeros(s-1).to(device),torch.ones(1).to(device)]).unsqueeze(1) for s in sessions_sizes]

        probs = [torch.solve(b, torch.t(Q))[0] for b,Q in zip(Bs,Qs) ]

        logProbs = [torch.log(p) for p in probs]

        loss = nn.NLLLoss()

        losses =  torch.stack([loss(slp.reshape(1,-1),y ) for slp,y in zip(logProbs,ys)])

        return torch.mean(losses),probs,logProbs




#collate_fn (callable, optional) – merges a list of samples to form a mini-batch.
#The batch argument is a list with all your samples. 
#in our setting, one sample is a full session. 
#we want to concatenate sessions
def my_collate(batch):
        # batch is a list of 6-element tuple
        # zip(*batch) unzips it: get a 6-element tuple of lists 
        #default_collate uses torch.stack, which concatenates along a new dimension
        #we want to concatenate them on dimension 0
        data = tuple(torch.cat(item) for item in zip(*batch))  

        ## first three items (label,ctxt_cat,ctxt_num) have length 1 for each session
        sessions_sizes = [item[3].shape[0] for item in batch] 
        
        return data[0], data[1:], sessions_sizes


class ChoiceDataset(Dataset):
    def __init__(self, data, ctxt_cat_cols=None, ctxt_num_cols=None,
                 alt_cat_cols=None, alt_num_cols=None, choice_col="choice"):
        """
        Characterizes a Dataset for PyTorch
    
        Parameters
        ----------
    
        data: list of pandas dataframes (one per session)
          each data frame object for the input data. It must
          contain all the continuous, categorical and the
          output columns to be used.
    
        *_cat_cols: List of strings
          The names of the categorical columns in the data.
          These columns will be passed through the embedding
          layers in the model. These columns must be
          label encoded beforehand. 
    
        *_num_cols: List of strings
          The names of the numerical columns in the data.
        
        choice_col: string.
          Name of the choice column.

        """
    
        #original data:list of dataframes,  one per session
        self.data = data
        

        self.ctxt_cat_cols = ctxt_cat_cols
        self.ctxt_num_cols = ctxt_num_cols
        self.alt_cat_cols = alt_cat_cols
        self.alt_num_cols = alt_num_cols

        self.choice_col = choice_col
    

    


    def __len__(self):
        """
        Denotes the total number of samples (sessions).
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Generates the pairs for the session indexed by idx
        """

        #take the df corresponfing to the given session
        df = self.data[idx]
        

        #context is taken from the first alternative of the session
        return (torch.LongTensor(np.where(df[self.choice_col].values==1)).contiguous(),
                torch.FloatTensor(df[self.ctxt_num_cols].iloc[[0]].values).contiguous(),
                torch.LongTensor(df[self.ctxt_cat_cols].iloc[[0]].values).contiguous(),
                torch.FloatTensor(df[self.alt_num_cols].values).contiguous(),
                torch.LongTensor(df[self.alt_cat_cols].values).contiguous(),
                )    
    
    
    


class FirstLayers(nn.Module):

    def __init__(self, emb_dims, no_of_cont, emb_dropout):
  
        """
        Parameters
        ----------
    
        emb_dims: List of two element tuples
          This list will contain a two element tuple for each
          categorical feature. The first element of a tuple will
          denote the number of unique values of the categorical
          feature. The second element will denote the embedding
          dimension to be used for that feature.
    
        no_of_cont: Integer
          The number of continuous features in the data.
    
    
        emb_dropout: Float
          The dropout to be used after the embedding layers.
    
        """
    
        super().__init__()
    
        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                         for x, y in emb_dims])
        
        
        self.no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_cont = no_of_cont
    
    
        # Dropout Layer
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
    
    def output_size(self):
        """
        Dimension of the output
        """
        return self.no_of_embs + self.no_of_cont 
  
  
    def forward(self, cont_data, cat_data):
  
        x = None
        
        if self.no_of_embs != 0:
            x = [emb_layer(cat_data[:, i])
                 for i,emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)
      
        if self.no_of_cont != 0:
      
            if x is not None:
                x = torch.cat([x, cont_data], 1) 
            else:
                x = cont_data
    
        return x
    
 
    
    
class FeedForwardNN(nn.Module):

    def __init__(self, input_size, lin_layer_sizes,
                 output_size, lin_layer_dropouts, activation):
  
        """
        Parameters
        ----------

        input_size : Integer 
          input dimension
    
        lin_layer_sizes: List of integers.
          The size of each linear layer. The length will be equal
          to the total number
          of linear layers in the network.
    
        output_size: Integer
          The size of the final output.
    
        lin_layer_dropouts: List of floats
          The dropouts to be used after each linear layer.

        activation:  Integer
          index of the desired activation function (0: ReLU, 1: Sigmoid, 2: Tanh, 3: LeakyReLU) 
        """
    
        super().__init__()
        
        if activation == 0:
            self.activation = nn.ReLU()
        elif activation == 1:
            self.activation = nn.Sigmoid()
        elif activation == 2:
            self.activation = nn.Tanh()
        elif activation == 3:
            self.activation = nn.LeakyReLU()

    
        # Linear Layers
        first_lin_layer = nn.Linear(input_size,
                                    lin_layer_sizes[0])
    
        self.lin_layers =\
         nn.ModuleList([first_lin_layer] +\
              [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
               for i in range(len(lin_layer_sizes) - 1)])
        
        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)
      
        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1],
                                      output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)
    
    
        # Dropout Layers
        self.droput_layers = nn.ModuleList([nn.Dropout(rate)
                                      for rate,size in zip(lin_layer_dropouts,lin_layer_sizes)])
    
    
    
    
    def forward(self, x):
  
        for lin_layer, dropout_layer in\
            zip(self.lin_layers, self.droput_layers):

            x = lin_layer(x)
        
            x = self.activation(x)

            x = dropout_layer(x)
      
        x = self.output_layer(x)
    
        #enforce positivity of transition rates and PCMC constraint : q_ij + q_ji > 0
        
        x = F.relu(x) + .5
    
        return x
    
    
class TrainableLayers(nn.Module):

    def __init__(self, ctxt_emb_dims, ctxt_no_of_cont, alt_emb_dims, alt_no_of_cont, lin_layer_sizes,
                 emb_dropout, lin_layer_dropouts, activation):
  
        """
        Parameters
        ----------
    
        emb_dims: List of two element tuples
          This list will contain a two element tuple for each
          categorical feature. The first element of a tuple will
          denote the number of unique values of the categorical
          feature. The second element will denote the embedding
          dimension to be used for that feature.
    
        no_of_cont: Integer
          The number of continuous features in the data.
    
        lin_layer_sizes: List of integers.
          The size of each linear layer. The length will be equal
          to the total number
          of linear layers in the network.
    
        output_size: Integer
          The size of the final output.
    
        emb_dropout: Float
          The dropout to be used after the embedding layers.
    
        lin_layer_dropouts: List of floats
          The dropouts to be used after each linear layer.

        activation: activation function to be used in the transition rate layers
        """
    
        super().__init__()
    
        # representation layer
        self.firstLayersCtxt = FirstLayers(ctxt_emb_dims, ctxt_no_of_cont,emb_dropout)
        self.firstLayersAlt = FirstLayers(alt_emb_dims, alt_no_of_cont,emb_dropout)

        input_size = self.firstLayersCtxt.output_size() + 2*self.firstLayersAlt.output_size()

        # transition rate layer
        self.FeedForwardNN = FeedForwardNN(input_size, lin_layer_sizes, 1, lin_layer_dropouts,activation)
        
    
 
    
    def forward(self, input_data,sessions_sizes,device): 
        ctxt_cont_data, ctxt_cat_data, alt_cont_data, alt_cat_data = input_data 

        ### representation layer

        ctxt = self.firstLayersCtxt(ctxt_cont_data, ctxt_cat_data)
        alt = self.firstLayersAlt(alt_cont_data, alt_cat_data)
        
        ### cartesian product and concatenation with context
        
        alt = torch.split(alt,sessions_sizes)
        
        ## this generates pairs in the order 11,12,23.. 21,22,23 ... 
        cart_prod = [torch.cat((torch.repeat_interleave(s,size, dim=0),s.repeat(size,1)), dim=1) for s,size in zip(alt,sessions_sizes)]

        if ctxt is not None:
            rep_ctxt = torch.repeat_interleave(ctxt,torch.pow(torch.tensor(sessions_sizes).to(device, non_blocking=True),2), dim=0)

            cart_prod = torch.cat([rep_ctxt, torch.cat(cart_prod)], dim=1)
        else:
            cart_prod = torch.cat(cart_prod)
        
        ### transition rate layer

        res = self.FeedForwardNN(cart_prod)
        
        return res 


    
