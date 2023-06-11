import numpy as np
from tqdm import tqdm
from channel import WirelessNetwork
import torch
from graph_utils import Model, GraphNeuralNetwork


mu_unconstrained = 0.01
step_size = 0.01
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Generator:
  def __init__(self,n,wx,wy,wc,d0=1,gamma=2.2,s=2,N0=1,device="cpu",batch_size=64,random=False):

    #Wireless Network configurations
    self.n = n
    self.wx = wx
    self.wy = wy
    self.wc = wc

    #Channel configurations
    self.d0 = d0
    self.gamma = gamma
    self.s = s
    self.N0 = N0

    #Training configurations
    self.device = device
    self.batch_size = batch_size

    #True if pathloss should change at random
    self.random=random

    self.train=None
    self.test=None

    #Generate a Wireless Network and pathloss matrix
    self.network = WirelessNetwork(self.wx,self.wy,self.wc,self.n,self.d0,self.gamma,self.s,self.N0)

    self.H1 = self.network.generate_pathloss_matrix()

  def __next__(self):
    if self.random:
      #Generate a new random network
      self.network = WirelessNetwork(self.wx,self.wy,self.wc,self.n,self.d0,self.gamma,self.s,self.N0)
      self.H1 = self.network.generate_pathloss_matrix()

    H2 = np.random.exponential(self.s,(self.batch_size,self.n,self.n))

    #Generate random channel matrix
    H = self.H1*H2

    #Normalization of the channel matrix
    eigenvalues,_ = np.linalg.eig(H)
    S = H/np.max(eigenvalues.real)

    #Put onto device
    H = torch.from_numpy(H).to(torch.float).to(self.device)
    S = torch.from_numpy(S).to(torch.float).to(self.device)
    return H, S, self.network
  
def train(model,update,generator,iterations):
    pbar = tqdm(range(iterations),desc=f"Training for n={generator.n}")

    for i in pbar:
      #For each iteration, generate a new random channel matrix
      H,S,network = next(generator)
      
      #Get the corresponding allocation strategy
      p = model(S)

      #Calculate the capacity as the performance under this allocation strategy
      c = network.generate_channel_capacity(p,H)

      #Update the parameters of the model
      update(p,c)

      pbar.set_postfix({'Capacity Mean': f"{c.mean().item():.3e}",
                        'Capacity Var': f"{c.var().item():.3e}",
                        'Power Mean': f"{p.mean().item():.3f}",
                        'Power Var': f"{p.var().item():.3f}"})


def test(model,generator,iterations):
    powers=[]
    capacities=[]
    loss=[]

    pbar = tqdm(range(iterations),desc=f"Test for n={generator.n}")
    for i in pbar:
      #For each iteration, generate a new random channel matrix
      H,S,network = next(generator)
      #Get the corresponding allocation strategy
      p = model(S)
      #Calculate the capacity as the performance under this allocation strategy
      c=network.generate_channel_capacity(p,H)

      #Save the loss,capacities and powers
      loss.append(-c.mean().item()+mu_unconstrained*p.mean().item())
      capacities.append(c.mean().item())
      powers.append(p.mean().item())
    
    print("Testing Results:")
    print(f"\tLoss mean: {np.mean(loss):4f}, variance {np.var(loss):.4f}"
          f"|Capacity mean: {np.mean(capacities):.4e}, variance {np.var(capacities):.4e}"
          f"|Power mean: {np.mean(powers): .4f}, variance {np.var(powers):.4f}")
  

if __name__ == "__main__":
    unconstrained = Model(GraphNeuralNetwork([5,5,5],[1,8,4,1]).to(device))
    optimizer = torch.optim.Adam(unconstrained.parameters(),step_size)

    N0 = 1e-6
    train_iterations = 200
    test_iterations = 100
    batch_size = 100

    def update_unconstrained(p,c):
        global mu, optimizer
        optimizer.zero_grad()
        objective = -c.mean()+mu_unconstrained*p.mean() #Specify loss function
        objective.backward()
        optimizer.step()

    generator_small = Generator(160,80,40,20,device=device,N0=N0,batch_size=batch_size)
    train(unconstrained,update_unconstrained,generator_small,train_iterations)  
    test(unconstrained,generator_small,test_iterations)

  




  