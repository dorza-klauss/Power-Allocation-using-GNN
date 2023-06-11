# Power-Allocation-using-GNN

In this project, we studied about GNN to solve the non-convex optimization problem of power allocation in an ad-hoc wireless network. 

We begin at first by building a model of point to point communication channel, which is then extended into a wireless network. A rayleigh fading model and the following parameter values are used in generating the network. 

|           Parameter         |  Value  |
|-----------------------------|--------:|
| Pathloss Reference distance |  1m     |
| Pathloss exponent           |  2.2m   |
| Fading energy               |  2m     |
| Noise floor                 |  1e-6mW |
| User density                |  0.05   |
| Receiver maximum distance   |  50m    |


A representation of a wireless network without the interference graph can be visulized in the figure below: 

![.](https://github.com/dorza-klauss/Power-Allocation-using-GNN/blob/main/plots/tx1.png)

The metric used to measure channel performance is the spectral efficiency, and the objective is to maximize the spectral efficiency by maximizing the SINR, which in turn is maximized by allocating power to each user. 

### Random Edge Graph Neural Network

We solve this non convex optimization problem by using a graph neural network to map the channel interference matrix **H** with power allocations **p(H)**. The inputs to the GNN is a graph with edges that come from a random distribution, and in this lab, the interference graph is the actual input, instead of the node state vector. Although the input and output layer has only one feature, but the intermediate hidden layers have multuple features and are processed by MIMO graph filters. The goal is to find the optimal filter tensors **A*** that maximises the power allocated to each user, which is now a function of the graph shift operator and the interference graph.

To train this as an emperical risk minimization problem, we define the loss function as the negative of the difference between the mean of the power allocated at each transmitter-receiver node, and the mean of the capacity of this allocation strategy at each node. Although this is suboptimal due to the non-convex nature of the problem, the results show that this helps us settle for a local optimal tensor **A***. 

### Stability and Transference

After training and generating the filter weights associated with each layer of the network, we test the well known observations of stability and transference of GNNs ourselves by generating a different wireless network with the same number of nodes and parameters, and evaluate the mean and variance of the GNN with the same set of filter coefficients. Similarly, for transference, we test the network with a larger network and different parameter values. 








