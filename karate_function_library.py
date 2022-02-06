import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def drawOriginalGraph(G):
    labels = [0 if G.nodes[idx]['club'] == 'Mr. Hi' else 1 for idx in range(len(G.nodes))]
    colors = ['yellow'if labels[idx] == 0 else 'orange' for idx in range(len(G.nodes))]
    pos = nx.drawing.layout.spring_layout(G)
    nx.drawing.draw(G,pos=pos,with_labels=True,node_color=colors)

#----------------------------------------------------------------------
def add_gaussian_features(G, means=[0.,.5], std=[1., 1.], nb_features=16):
    # Different randomization for each run
    # Create additional attributes `node_metadata`, `labels`, `nb_nodes`
    # Note that networkx has its own mechanism to handle attributes, but
    # I am operating outside this mechanism

    nodes = np.asarray(G.nodes)
    nb_nodes = nodes.shape[0]
    node_metadata = np.random.randn(nb_nodes, nb_features)
    labels = np.asarray([0 if G.nodes[node]['club'] == 'Mr. Hi' else 1 for node in list(G.nodes)])
    mean = np.where(labels == 0, means[0], means[1])
    std = np.where(labels == 0, std[0], std[1])
    node_metadata = mean.reshape(-1, 1) + std.reshape(-1,1) * node_metadata

    # Add labels and metadata to the networkx graph `G`
    G.node_metadata = node_metadata
    G.labels = labels
    G.nb_nodes = nb_nodes

#----------------------------------------------------------------------
def plot_metadata(G):
        indices = np.asarray(range(G.nb_nodes))
        indices_0 = indices[G.labels == 0]
        indices_1 = indices[G.labels == 1]
        features_0 = G.node_metadata[indices_0, :].reshape(-1)
        features_1 = G.node_metadata[indices_1, :].reshape(-1)
        plt.hist(features_0, color='green', histtype='step', label='label 0')
        plt.hist(features_1, color='blue', histtype='step', label='label 1');
        plt.title("Features as normal distributions")
        plt.xlabel("Feature value")
        plt.ylabel("Histogram")
        plt.legend()

#----------------------------------------------------------------------
