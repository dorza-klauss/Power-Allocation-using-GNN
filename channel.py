import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from tqdm import tqdm
from typing import List, Union, Tuple
import torch
from torch import nn

class Channel(object):
  def __init__(self,d0,gamma,s,N0):
    """
    Object for modeling a channel

    Args:
    d0: Reference dislstance
    gamma: Path loss exponent
    s: Fading energy
    N0: Noise floor

    """


    self.d0 = d0
    self.gamma = gamma
    self.s = s
    self.N0 = N0

  def pathloss(self,d):

    """
    Calculates the simplified path loss value

    Args: 
    d: Distance, which can be a matrix.

    """

    return (self.d0/d)**self.gamma #element wise operation

  def fading_channel(self,d,Q):
    """
    Calculates the fading channel model.

    Args: 
    d: Distance
    Q: Number of random samples

    Returns Q fading channel realizations

    """

    Exp_h = (self.d0/d)**self.gamma
    h_til = np.random.exponential(self.s,size=(1,Q))
    h = Exp_h*h_til/self.s
    return h

  def build_fading_capacity_channel(self,h,p):

    """
    Calculates the fading channel capacity channel model

    Args: 
    h: Fading channel realizations (of length Q)
    p: Power values (of length Q)

    Returns Q channel capacity values

    """

    return np.log(1+h*p/self.N0)
  

class WirelessNetwork(object):

  def __init__(self,wx,wy,wc,n,d0,gamma,s,N0):

    """
    Object for modelling a wireless network

    Args:
    wx: Length of area
    wy: Width of area
    wc: Max distances of receiver from transmitter
    n: Number of transmitters/receivers
    d0: Reference distance
    gamma: Pathloss exponent
    s: Fading energy
    N0: Noise floor

    """

    self.wx = wx
    self.wy = wy
    self.wc = wc
    self.n = n

    #Determine transmitter and receiver positions
    self.t_pos, self.r_pos = self.determine_positions()

    #Calculate distance matrix using scipy,spatial method

    self.dist_mat = distance_matrix(self.t_pos, self.r_pos)

    self.d0 = d0
    self.gamma =gamma
    self.s = s
    self.N0 = N0

    #Creates a channel with the given parameters
    self.channel = Channel(self.d0,self.gamma,self.s,self.N0)

  
  def determine_positions(self):
    """
    Calculate positions of tranmitters and receivers

    """

    #Calculate transmitter positions
    t_x_pos = np.random.uniform(0,self.wx,(self.n,1))
    t_y_pos = np.random.uniform(0,self.wy,(self.n,1))
    t_pos = np.hstack((t_x_pos,t_y_pos))

    #Calculate receiver positions
    r_distance = np.random.uniform(0,self.wc,(self.n,1))
    r_angle = np.random.uniform(0,2*np.pi,(self.n,1))
    r_rel_pos = r_distance*(np.hstack((np.cos(r_angle),np.sin(r_angle))))
    r_pos = t_pos+ r_rel_pos
    return t_pos, r_pos

  def generate_pathloss_matrix(self):

    """  
    Calculates the pathloss matrix from the distance matrix
    """ 
    
    return self.channel.pathloss(self.dist_mat)

  
  def generate_interference_graph(self,Q):

    """
    Calculates interference graph  using the fading_channel function

    """

    return self.channel.fading_channel(self.dist_mat,Q)

  def generate_channel_capacity(self,p,H):

    """
    Calculates capacity of each transmitter


    Extracting the diagonals from the 2d matrices made up by the last two dimensions of H, where H is a tensor with dimensions > 2, and we are 
    mainly interested in the last two dimensions size_n, size_n to get the diagonal elements
   
    """
    num = torch.diagonal(H, dim1=-2,dim2=-1)*p #required signal
    den = H.matmul(p.unsqueeze(-1)).squeeze()-num + self.N0 #interference+noise
    return torch.log(1+num/den)

  def plot_network(self):

    """
    Creates a plot of the given Wireless Network

    """

    plt.scatter(self.t_pos[:,0], self.t_pos[:,1], s=4, label="Transmitters")
    plt.scatter(self.r_pos[:,0], self.r_pos[:,1], s=4, label="Receivers")
    plt.xlabel("Area Length")
    plt.ylabel("Area Width")
    plt.title("Wireless Network")
    plt.legend()
    return plt.show()
    

def pathloss_test():
    #Create distance from 1-100
    dist = np.arange(1,101)

    #Create channel object 
    channel = Channel(1,2.2,2,1e-6)

    #Calculae 100 pathloss values
    Exp_h = channel.pathloss(dist)

    #Plot in linear scale
    plt.figure()
    plt.plot(Exp_h)
    plt.ylabel('Pathloss')
    plt.xlabel('Distance')
    plt.title('Pathloss vs Distance')
    plt.show()



    #Plot in logarithmic scale
    plt.figure()
    plt.plot(np.log(dist), np.log(Exp_h))
    plt.ylabel('Pathloss')
    plt.xlabel('Distance')
    plt.title('Pathloss vs Distance')
    plt.show()



    #Fading channel

    #Testing the plot with distance ranging from 1 to 100 m
    #100 channel realizations
    Q = 100

    #Initiate ta matrix to store results, 100 is for distance 1 to 100
    h_sim = np.zeros((100,Q))

    #Consider distances from 1-100 meters
    #and compute 100 realizations of at each 

    for d in dist: 
        h_sim[d-1,:] = channel.fading_channel(d,100)

    #each row has 100 different channel realizations for a fixed distance

    #Calculate mean and var of these realizations
    h_mean = np.mean(h_sim,axis=1)
    h_var = np.var(h_sim,axis=1)

    # Plot
    plt.figure()
    plt.errorbar(dist, h_mean, h_var, ecolor='orange')
    plt.ylabel('h')
    plt.xlabel('Distance')
    plt.title("Fading Channel Samples vs. Distance")
    plt.show()
  
    #Capacity plots


    #using h_sim to calculate Q channel capacity values
    cap = channel.build_fading_capacity_channel(h_sim,0.05)

    #calculate mean and var of these realizations
    cap_mean = np.mean(cap,axis=1)
    cap_var = np.var(cap,axis=1)

    # Plot
    plt.figure()
    plt.errorbar(dist, cap_mean, cap_var, ecolor="orange")
    plt.ylabel('Capacity')
    plt.xlabel('Distance')
    plt.title('Channel Capacity vs. Distance')
    plt.show()

def test(): 
    d0 = 1
    gamma = 2.2
    s = 2
    N0 = 1e-6
    rho = 0.05
    wc = 50
    wx = 200
    wy = 100
    n = int(rho*wx*wy) #from transmitter density
    WirelessNetwork(wx,wy,50,n,d0,gamma,s,N0).plot_network()

if __name__ == "__main__":
    test()

    