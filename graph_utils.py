import torch
import torch.nn as nn
import numpy as np
from typing import List, Union, Tuple


device = "cuda:0" if torch.cuda.is_available() else "cpu"

class GraphFilter(nn.Module):
  def __init__(self,k:int,f_in=1,f_out=1,f_edge=1):
    """
    A graph filter layer.

    Args:
      gso: Graph shift operator
      k: The number of filter taps
      f_in: The number of input features
      f_out: The number of output features

    """

    super().__init__()
    self.k = k
    self.f_in = f_in
    self.f_out = f_out
    self.f_edge = f_edge

    self.weight = nn.Parameter(torch.ones(self.f_out,self.f_edge,self.k,self.f_in))
    self.bias = nn.Parameter(torch.zeros(self.f_out,1))
    torch.nn.init.normal_(self.weight,0.3,0.1)
    torch.nn.init.zeros_(self.bias)

  def to(self,*args,**kwargs):
    #Only the filter taps and weights are registered as parameters, so we move the gsos ourselves
    self.weight = self.gso.to(*args,**kwargs)
    self.bias = self.gso.to(*args,**kwargs)
    return self

  def forward(self,x:torch.Tensor,S:torch.Tensor):

    batch_size = x.shape[0]

    B = batch_size
    E = self.f_edge
    F = self.f_out
    G = self.f_in
    N = S.shape[-1] #Number of nodes
    K = self.k #Number of filter taps

    h = self.weight
    b = self.bias

    #x in BxGxN and S in ExNxN
    #We need to come up with a matrix multiplication that yields z = x*S with shape
    #BxExKxGxN.

    #For this, we add the corresponding dimensions
    x = x.reshape([B,1,G,N])
    S = S.reshape([batch_size,E,N,N])
    z = x.reshape([B,1,1,G,N]).repeat(1,E,1,1,1) #This is for k=0
    #We need to repeat along the E dimension, because for k=0, S_{e}=l for all e,
    #therefore, the same signal values have to be used along all edge feature dimensions

    for k in range(1,K):
      x = torch.matmul(x,S) #BxExGxN
      xS = x.reshape([B,E,1,G,N]) #BxEx1xGxN
      z = torch.cat((z,xS),dim=2) #BxExKxGxN

    #The output z is of size BxExKxGxN
    #Now we have the x*S_{e}^{k} product, and we need to multiply with the filter taps.
    #We multiply z on the left, and h on the right, the ouput is to be BxNxF(the multiplication is not along the N dimension),
    #so we reshape z to be BxNxExKxG and reshape it to BxNxEKG (we always reshape the last dimesnions),
    #and then make h be ExKxGxF and reshape it to EKGxF, and then multiply

    y = torch.matmul(z.permute(0,4,1,2,3).reshape([B,N,E*K*G]), h.reshape([F,E*K*G]).permute(1,0)).permute(0,2,1)

    #And permute again to bring it from BxNxF to BxFxN
    #Finally add the bias

    if b is not None:
      y = y+b
    return y

class GraphNeuralNetwork(nn.Module):
  def __init__(self,ks:Union[List[int], Tuple[int]] = (5,), 
               fs: Union[List[int],Tuple[int]]=(1,1)):

    """
    An L layer graph neural network. Uses ReLU activation for each layer except the last, which has no activation

    Args:
      gso: Graph shift operator
      ks: [K_1,...K_L] On ith layer, K_{i} is the number of filter taps
      fs: [F_1,...F_L]. On the ith layer, F_{i} and F{i+1} are the number of input and output features

    """

    super().__init__()
    self.n_layers = len(ks)

    self.layers = []
    for i in range(self.n_layers):
      f_in = fs[i]
      f_out = fs[i+1]
      k = ks[i]
      gfl = GraphFilter(k,f_in,f_out)
      activation = torch.nn.ReLU() if i<self.n_layers-1 else torch.nn.Identity()
      self.layers += [gfl,activation]
      self.add_module(f"gfl{i}", gfl)
      self.add_module(f"activation{i}",activation)

  def forward(self,x,S):
    for i, layer in enumerate(self.layers):
      x = layer(x,S) if i%2==0 else layer(x)
    return x

class Model(torch.nn.Module):
  def __init__(self,gnn):
    super().__init__()
    self.gnn = gnn

  def forward(self,S):
    batch_size = S.shape[0]
    n = S.shape[1]
    p0 = torch.ones(batch_size,n,device=device)
    p = self.gnn(p0,S).abs()
    return torch.squeeze(p)