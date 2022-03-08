import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.sparse as tsp

def drawOriginalGraph(G):
    labels = [0 if G.nodes[idx]['club'] == 'Mr. Hi' else 1 for idx in range(len(G.nodes))]
    colors = ['yellow'if labels[idx] == 0 else 'orange' for idx in range(len(G.nodes))]
    pos = nx.drawing.layout.spring_layout(G)
    nx.drawing.draw(G,pos=pos,with_labels=True,node_color=colors)

#----------------------------------------------------------------------
def add_gaussian_features(G, node_means=[0.,.5], node_stds=[1., 1.], edge_means=[-1.,3.], edge_stds=[1.,1.], 
        nb_node_features=16, nb_edge_features = 8):
    # Different randomization for each run
    # Create additional attributes `node_metadata`, `labels`, `nb_nodes`
    # Note that networkx has its own mechanism to handle attributes, but
    # I am operating outside this mechanism

    nodes = np.asarray(G.nodes)
    edges = np.asarray(G.edges)
    nb_nodes = nodes.shape[0]
    nb_edges = edges.shape[0]
    node_metadata = np.random.randn(nb_nodes, nb_node_features)
    edge_metadata = np.random.randn(nb_edges, nb_edge_features)

    # Labels are only defined on nodes (not on edges)
    n_labels = np.asarray([0 if G.nodes[node]['club'] == 'Mr. Hi' else 1 for node in list(G.nodes)])
    e_labels = np.random.randint(0, 2, nb_edges) # random 0 and 1

    # Node metadata
    n_means = np.where(n_labels == 0, node_means[0], node_means[1])
    n_stds = np.where(n_labels == 0, node_stds[0], node_stds[1])
    node_metadata = n_means.reshape(-1, 1) + n_stds.reshape(-1,1) * node_metadata

    # Edge metadata
    # Need edge labels to decide which features to use. So create some edge labels
    e_means = np.where(e_labels == 0, edge_means[0], edge_means[1])
    e_stds = np.where(e_labels == 0, edge_stds[0], edge_stds[1])
    edge_metadata = e_means.reshape(-1, 1) + e_stds.reshape(-1,1) * edge_metadata

    # Node and edge metadata
    G.node_features = torch.tensor(node_metadata, requires_grad=False).float()
    G.edge_features = torch.tensor(edge_metadata, requires_grad=False).float()

    # Add labels to the networkx graph `G`
    G.n_labels = G.labels = torch.tensor(n_labels, requires_grad=False).float()
    print(G.D)
    G.e_labels = torch.tensor(e_labels, requires_grad=False).float()
    G.nb_nodes = nb_nodes
    G.nb_edges = nb_edges
    G.nb_node_features = nb_node_features
    G.nb_edge_features = nb_edge_features
    G.nb_graph_features = 0

#--------------------------------------------------------------------------
def update_associated_matrices(G):
    # wasteful of memory but ok for small graphs (N < 1000)
    G.A = torch.tensor(nx.linalg.graphmatrix.adjacency_matrix(G).toarray()).detach().float()
    G.I = torch.eye(G.nb_nodes)
    #G.L = G.D - G.A  # Laplacian
    G.An = G.I + G.A
    G.D = torch.sum(G.An, dim=0)  # degree matrix (list)
    G.Dinvsq = torch.diag(np.sqrt(1.0 / G.D))  # matrix
    G.Dinv = torch.diag(G.D)  # matrix
    G.An = torch.tensor(G.Dinvsq @ G.An @ G.Dinvsq).float() # symmetric normalization

    # Store sparse representaiton of G.Dinv (diagonal matrix)
    indexDinv = torch.tensor([list(range(0, G.nb_nodes)), list(range(0, G.nb_nodes))])
    VDinv = torch.diag(G.Dinv)
    G.DinvCoo = torch.sparse_coo_tensor(indexDinv, VDinv, [G.nb_nodes, G.nb_nodes])
    

    # Create B matrix: Ne x N in coo format
    indexB = torch.tensor(np.asarray(G.edges).T)
    VB = torch.ones(G.nb_edges)
    G.B = torch.sparse_coo_tensor(indexB, VB, [G.nb_edges, G.nb_nodes])
    G.Btransp = G.B.transpose(0, 1)
    print("G.B: ", G.B.shape, "G.B.transp: ", G.Btransp.shape)
    #print("GBT: ", G.Btransp)
    #print(G.edges)

#----------------------------------------------------------------------
def plot_metadata(G):
        indices = np.asarray(range(G.nb_nodes))
        indices_0 = indices[G.labels < 0.5] # labels are floats
        indices_1 = indices[G.labels > 0.5]
        features_0 = G.node_features[indices_0, :].reshape(-1)
        features_1 = G.node_features[indices_1, :].reshape(-1)
        plt.hist(features_0.numpy(), color='green', histtype='step', label='label 0')
        plt.hist(features_1.numpy(), color='blue', histtype='step', label='label 1');
        plt.title("Features as normal distributions")
        plt.xlabel("Feature value")
        plt.ylabel("Histogram")
        plt.legend()

#----------------------------------------------------------------------
#--------------------------------------------------------------------------
