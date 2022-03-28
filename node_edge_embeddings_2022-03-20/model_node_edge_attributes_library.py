import torch
from torch import tensor
from collections import defaultdict
import numpy as np

def set_weight(n_in, n_out):
    W = torch.zeros(n_in, n_out, dtype=torch.float, requires_grad=True)
    # Check out Xavier Initialization
    # https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
    rmax = 1. / n_in ** 0.5
    torch.nn.init.uniform_(W, -rmax, rmax)
    W = torch.nn.Parameter(W)
    return W


class GNNNodesEdges(torch.nn.Module):
    # the graph G will contain the adjacency matrix A, and the node-edge matrix B and Btransp. 
    # Saving B and Btransp is inefficient but convenient
    def __init__(self, G, ignore_edges=False):
        super(GNNNodesEdges, self).__init__()

        self.G = G
        self.d = self.nb_node_features = G.nb_node_features
        self.f = self.nb_edge_features = G.nb_edge_features
        self.B = G.B
        self.Btransp = G.Btransp
        self.DinvCoo = G.DinvCoo
        self.ignore_edges = ignore_edges
   
        # One question: why not simply use self.relu1 for all three relu nodes? 
        self.relu1 = torch.nn.ReLU();
        self.relu2 = torch.nn.ReLU();
        self.relu3 = torch.nn.ReLU()
        
        # Note that I could choose to make W0 and W2 the same to save memory. 
        # Whether that would help or not would have to be tested. 
        self.W0 = set_weight(self.f, self.f)
        self.W1 = set_weight(self.d, self.f)
        self.W2 = set_weight(self.f, self.f)
        self.W3 = set_weight(self.f, self.d)
        # Alpha is a scalar variable
        self.alpha = set_weight(1, 1)
                
    def forward(self, X0, E0):
        """
        Parameters:
        -----------
            V0 : torch tensor
                feature matrix |V| x nb_node_features
            E0 : torch tensor
                feature matrix |E| x nb_edge_features
        return:
        -------
            return tuple: updated (node_embeddings, edge_embeddings)
            
        Embeddings at the zero'th iteration are set to the features
        """
        E  = self.relu1(self.update_edges(X0, E0))
        EX = self.relu2(self.edges_to_nodes(X0, E))
        X  = self.relu3(self.update_nodes(X0, EX))
        return X, E
        
    def update_edges(self, X, E):
        if self.ignore_edges:
            return (self.B @ X) @ self.W1
        else:
            return (self.B @ X) @ self.W1 + E @ self.W0
    
    def edges_to_nodes(self, X, E):
        # normalized by Dinv
        # @ only works with sparse multiplied by dense
        return self.DinvCoo @ (self.Btransp @ (E @ self.W2))
    
    def update_nodes(self, X, EX):
        return self.alpha * X + EX @ self.W3

    #----------------------------------------------------------------------

class myGCN(torch.nn.Module):
    def __init__(self, G, ignore_edges=False):
        super(myGCN, self).__init__()
        self.G = G
        
        # self.gcn1 = GCNConv(G, n_in, n_in//4)
        self.gcn1 = GNNNodesEdges(G, ignore_edges)

        # self.gcn1 = GCNConv(G, n_in, n_in//2)
        # self.gcn2 = GCNConv(G, n_in//2, n_in//4)
        n_in = G.nb_node_features
        self.linear1 = torch.nn.Linear(n_in, n_in//4)
        self.linear2 = torch.nn.Linear(n_in//4, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, X0, E0):
        """
        Parameters:
        -----------
            H0 : torch tensor
                feature matrix |V| x nb_features
        return:
        -------
            Layout output
        """
        X, E = self.gcn1(X0, E0)

        # Save norm of node/edge embeddings on original graph, same dims as X0, E0
        self.G.node_embeddings_list.append(X.norm())
        self.G.edge_embeddings_list.append(E.norm())

        #X, E = self.gcn1(X, E)  # apply GNN twice (does not work)
        X = self.linear1(X)
        X = self.sigmoid(self.linear2(X))

        return X

    #-----------------------------------------------------------------

class BinaryCrossEntropyLoss:
    def __init__(self, mask):
        self.mask = mask
        
    def __call__(self, hidden, target):
        H = hidden
        Y = target
        """
        summation over all edges
        Training set: defined by mask[i] = 0
        H[i,0] is the probability of Y[i] == 1 (target probability)
        H[i,1] is the probability of Y[i] == 0 (target probability)

        Parameters
        ----------
        target : torch.tensor of shape [nb_nodes, 2]


        Y : torch.tensor of shape [nb_nodes]
            labels

        Notes 
        -----
        The target must be in the range [0,1].
        """
        costf = 0
            
        for i in range(Y.shape[0]):
            if self.mask[i] == 0:  # training set
                costf -= (Y[i] * torch.log(H[i,0]) + (1-Y[i]) * torch.log(1.-H[i,0]))

        return costf

    #------------------------------------------------------------------

    #--------------------------------------------------------------

def new_train(G, model, mask, loss_fn, optimizer, nb_epochs):
    X0 = torch.tensor(G.node_features).float()
    E0 = torch.tensor(G.edge_features).float()
    labels = torch.tensor(G.labels, requires_grad=False)
    losses = []
    node_embeddings = []
    edge_embeddings = []
    accuracy_count = defaultdict(list)
    G.node_embeddings_list = []
    G.edge_embeddings_list = []

    for epoch in range(nb_epochs):
        model.train()
        optimizer.zero_grad()
        print("X0.norm: ", X0.norm())
        pred = model(X0, E0)
        loss = loss_fn(pred, labels)
        if np.isnan(loss.detach()):
            break
        losses.append(loss.item())

        with torch.no_grad():  # should not be necessary
            loss.backward(retain_graph=False)
            optimizer.step()

        model.eval()
        predict(model, G, mask, accuracy_count)

    return losses, accuracy_count, node_embeddings, edge_embeddings

    #--------------------------------------------------------------

def predict(model, G, mask, accuracy_count):

    X0 = torch.tensor(G.node_features).float()
    E0 = torch.tensor(G.edge_features).float()
    Y = G.labels

    # Follow https://www.analyticsvidhya.com/blog/2021/08/linear-regression-and-gradient-descent-in-pytorch/
    H = model(X0, E0)

    count_correct = [0,0]
    count = [0,0]
    for i in range(H.shape[0]):
        if mask[i] == 1: # test data
            count[1] += 1
            if H[i] > 0.5 and Y[i] > 0.9:
                count_correct[1] += 1
            if H[i] < 0.5 and Y[i] < 0.1:
                count_correct[1] += 1
        else:  # mask == 0, training data
            count[0] += 1
            if H[i] > 0.5 and Y[i] > 0.9:
                count_correct[0] += 1
            if H[i] < 0.5 and Y[i] < 0.1:
                count_correct[0] += 1

    if count[0] != 0 and count[1] != 0:
        accuracy_count['train'].append(count_correct[0] / count[0])
        accuracy_count['test'].append(count_correct[1] / count[1])
    else:
        accuracy_count['train'].append(0)
        accuracy_count['test'].append(0)

    #--------------------------------------------------------------- 
