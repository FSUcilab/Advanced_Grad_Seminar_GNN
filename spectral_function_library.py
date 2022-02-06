import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy 

def compute_total_variation(m, eigs, eigv):
    # Global Variation of matrix, acting as a shift operator
    # Calculate v - mm @ v and compute L1 norm where v is an eigenvalue
    max_eigenval = np.max(np.abs(eigs))
    mm = m / max_eigenval
    ss_norms = []
    ss_eigvec = np.array(eigv)

    for ix, eig in enumerate(eigs):
        eigvec = eigv[:, ix]
        ss_eigvec[:, ix] = eigvec - mm @ eigvec
        ss_norm = np.linalg.norm(ss_eigvec[:, ix], ord=1)
        ss_norms.append(ss_norm)

        def check_details(m, mm, eig, eigvec):
            # Compare mm*eigvec against eigs[:]*eigvec
            mmeigvec = mm @ eigvec
            eigeigvec = eig * eigvec
            meigvec = m @ eigvec
            print("==> mmeigvec / eigeigvec: ", mmeigvec / eigeigvec)
            print(
                "    meigvec / eigeigvec: ", meigvec / eigeigvec
            )  # ratio of 1 across all eigenvectors
            print("    eigvec*eigvec: ", np.dot(eigvec, eigvec))  # unit vector
            print("    eigenvalue: ", eig)
            print("    mmeigvec: ", mmeigvec)
            print("    eigeigvec: ", eigeigvec)
            print("    meigvec: ", meigvec)

        # print detailed calculations
        # check_details(m, mm, eig, eigvec)
    return eigs, ss_eigvec, ss_norms
#------------------------------------------------------------------
def generate_eigenvectors_from_adjacency_matrix_1(G, N, seed):
    """
    Arguments
    N: number of nodes
    """
    np.random.seed(seed)

    # Convert to np.ndArray
    A = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()

    # Different matrices
    D = np.sum(A, axis=0)
    D = np.diag(D)
    L = D - A
    Lapl = nx.linalg.laplacianmatrix.laplacian_matrix(G).toarray()
    invD = np.linalg.inv(D)
    invDA = A * invD
    invDL = invD * L
    invDLinvD = np.sqrt(invD) * L * np.sqrt(invD)

    Ln = nx.normalized_laplacian_matrix(G)
    Ln = Ln.toarray()  # from sparse array to ndarray

    nb_rows, nb_cols = 5, 3
    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(10, 12))

    matrix = ["A", "D", "L", "Lapl", "invD", "invDA", "invDL", "invDinvD", "Ln"]
    matrices = [A, D, L, Lapl, invD, invDA, invDL, invDLinvD, Ln]
    matrix = ["A", "L"]
    matrices = [A, L, invDL, invDLinvD, Ln]
    matrix = ["A", "L", "$D^{-1}L$", "$D^{-1/2}LD^{-1/2}$", "Ln"]
    nb_matrices = len(matrices)
    # ax_graph = axes[2 * nb_matrices]
    # ax_eigenfct = axes[nb_matrices + 1]

    # Eigenvalues
    fig.suptitle("Eigenvalues of various matrices")
    for i, m in enumerate(matrices):
        ax = axes[i, 0]
        eigen = np.linalg.eig(m)
        eigs, eigv = eigen
        args = np.argsort(eigs)[::-1]
        eigs, eigv = eigs[args], eigv[:, args]

        sorted_eigs, ss_eigvec, ss_norms = compute_total_variation(m, eigs, eigv)

        # plot eigenvalues and the Total variation
        ax.set_title(matrix[i])
        ax.grid(True)
        ax.plot(sorted_eigs, "-o", label="$\lambda_i$")
        ax.plot(ss_norms, "-x", label="Total Variation")
        ax.legend()

        # plot eigenfunctions
        ax = axes[i, 1]
        for j in range(4):
            ax.plot(eigv[:,j], label="$v_%d$" % j)
        ax.set_title("Eigenfunctions(%s)" % matrix[i])
        if (i == nb_matrices-1):
            ax.legend()

        # plot shifted eigenfunctions
        ax = axes[i, 2]
        for j in range(4):
            ax.plot(ss_eigvec[:,j], label="$shifted(v_%d)$" % j)
        ax.set_title("shifted eigvct(%s)" % matrix[i])
        if (i == nb_matrices-1):
            ax.legend()

    # Draw the graph in the last subplot
    plt.subplot(nb_rows, nb_cols, nb_rows*nb_cols)
    nx.draw(G)

    for i in range(i + 2, axes.shape[-1]):
        axes[i, 0].axis("off")

    plt.tight_layout()
    plt.show()
#----------------------------------------------------------
def degree_matrix(A):
    # Can return non-invertible matrices when graph is directed
    return np.diag(np.sum(A, axis=0))
#--------------------------------------------------------------------------
def normalized_matrix(Amatrix, Dmatrix, type_norm):

    if type_norm == "left":
        Dinv = np.diag(1. / np.diag(Dmatrix)) # efficient, returns matrix
        return Dinv @ Amatrix
    elif type_norm == "symmetric":
        Dinvsq = np.diag(1. / np.sqrt(np.diag(Dmatrix))) # efficient, returns matrix
        return Dinvsq @ Amatrix @ Dinvsq
    else: # no normalization
        return Amatrix
#-------------------------------------------------------------------------
def linear_acyclic_chain(N, graph_type):
    A = np.zeros([N,N])
    for i in range(1, N):
        A[i, i - 1] = 1
    if graph_type == "undirected":
        A = A + A.T
    return A
#------------------------------------------------------------------------
def tot_var(A, v):
    """
    Calculate the total variation: \sum_i (s[i]-s[j])^2
    where s is a signal, which could be an eigenvector of $A$.
    The function is inefficient but will work on general graphs. 
    
    Parameters
    ----------
    A : Numpy array
        Adjacency Matrix
    v : List
        Graph signal
    
    Returns
    -------
    A list with the total variation of the signal `v` for each eigenvalue.
    """
    total_variat = 0
    N = len(v)
    for i in range(N):
        for j in range(i):
            if abs(A[i, j]) > 0.01:
                total_variat += (v[i] - v[j]) ** 2
    return total_variat
#-------------------------------------------------------------------
def plot_one_curve(
    ax,
    curve,
    xlabel="[Add x-label]",
    ylabel="[Add y-label]",
    title="[Add title]",
    style="-o",
    color="black",
    xlim=None,
    ylim=None,
    ms=None,  # symbol scaling factor
    label=None
):
    """
    Plots a single curve using matplotlib. 
    
    Parameters
    ----------
    ax : plot axis returned by subplots or gca()
    curve: 1-D numpy array with the curve data:w
    
    
    Returns
    -------
    No returns
    """
    ax.plot(curve, style, color=color)
    ax.grid(True)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
#-------------------------------------------------------------------
def plot_multi_curves(
    ax,
    x, 
    curves,
    xlabel="[Add x-label]",
    ylabel="[Add y-label]",
    title="[Add title]",
    style="-o",
    xlim=None,
    ylim=None,
    ms=None,  # symbol scaling factor
    labels=None
):
    """
    Plots multiple curves using matplotlib. 
    
    Parameters
    ----------
    ax : plot axis returned by subplots or gca()
    curves: 2-D numpy array
        The first dimension is the number of points; the second
        dimension holds the curve data values. 
    
    Returns
    -------
    No returns
    """
    
    print("gordon")
    assert(curves.shape[1] == len(labels))
    #print("enter plot_multi_curves")
    #print("labels: ", labels)
    #print("curve shape[1]: ", curve.shape[1])

    nb_curves = curves.shape[1]
    for k in range(nb_curves):
        #print("label: ", labels[k])
        ax.plot(x, curves[:,k], style, label=labels[k])
        #ax.plot(curve, style, label=f"$gor_{k}$")

    ax.grid(True)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if labels:
        ax.legend(framealpha=0.5, fontsize=8)
#-------------------------------------------------------------------
def plot_data1(H_dict, eigval_dict, eigvec_dict, totvar, which_eig):
    """
    This function plots eigenvalues, eigenvectors, the total variation 
    based on the Graph Laplacian, and a single eigenfunction corresponding 
    to the eigenvalue `which_eig`.  
    
    Parameters
    ---------
    H_dict : dictionary 
        Keys are normalization types ('none', 'left', 'symmetric'). 
    eigenval_dict: dictionary with integer keys
        Keys are the eigenvalue index. 
        The eigenvalue spectrum is eigenval_dict[key][:]
    eigvec_dict: dictionary 
        Keys are the normalization types. 
        The kth eigenfunction  eigenvec_dict[key][:, k]
    totvar: list
        Total variation $\sum_i A_{i,j} (s_i - s_j)^2$
    which_eig: int
        Display an eigenfunction corresponding to eigenvalue `which_eig`
        
    Returns
    -------
    There are no returns.
    """
    N = list(H_dict.values())[0].shape[0]
    nrows = 3
    ncols = 3
    # rows and cols are used to access axes array elements
    row_eigf, row_eigv = 0, 1
    cols_dict = {"none": 0, "left": 1, "symmetric": 2}
    pos_none_eig = 2, 1
    pos_none_tot_var = 2, 0

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6))
    #print("H_dict keys: ", list(H_dict.keys()))

    for k, v in H_dict.items():
        eigval_dict[k], eigvec_dict[k] = np.linalg.eig(v)
        arg = np.argsort(eigval_dict[k])
        eigval_dict[k] = eigval_dict[k][arg]
        eigvec_dict[k] = eigvec_dict[k][:, arg]

    # Loop over matrix types
    for k in H_dict.keys():
        # Eigenvectors 
        nb_fcts = 5   # can be changed
        ax = axes[row_eigf, cols_dict[k]]
        plot_multi_curves(
            ax,
            eigvec_dict[k][:, 0:nb_fcts],
            style="-o",
            labels=[f"$\lambda_{i}$" for i in range(0,nb_fcts)],
            title="Eigenfunctions",
            xlabel="k",
            ylabel="v_k",
            ylim=[-0.2, 0.2]
        )

        # Eigenvalues
        ax = axes[row_eigv, cols_dict[k]]
        plot_one_curve(
            ax,
            eigval_dict[k],
            style="-o",
            xlabel="k",
            title="Eigenvalues",
            ylabel="$\lambda_k$",
            ylim=[0, 5]
        )

    ax = axes[pos_none_eig]
    ax.set_ylim(-0.2, 0.2)
    ax.grid(True)
    ax.set_title("Single Eigenvector, no normalization")

    try:
        eigvec = eigvec_dict["none"][:, which_eig]
    except:
        print(f"which_eig must be < N! Reset value to ${N-1}$")
        which_eig = N - 1
        eigvec = eigvec_dict["none"][:, which_eig]

    plot_one_curve(
        ax,
        eigvec,
        style="-o",
        xlabel="k",
        ylabel="v_k",
        color="black",
        label=f"$\lambda_{which_eig}$",
    )

    ax = axes[row_eigv, cols_dict["none"]]
    ax.plot(which_eig, eigval_dict["none"][which_eig], "o", ms=10, color="red")
    ax.set_title(f"Eigenvalues $\lambda_k$")

    ax = axes[pos_none_tot_var]

    plot_one_curve(ax, totvar, title="Total Variation, $L$, no normalization")
    ax.plot(which_eig, totvar[which_eig], "o", ms=10, color="red")

    for k in H_dict.keys():
        ax = axes[0, cols_dict[k]]
        ax.set_title("Normalization: " + k)
        ax = axes[1, cols_dict[k]]
        ax.set_title("Normalization: " + k)

    plt.suptitle(
        "Eigenvectors and eigenvalues for $L$ (left), $D^{-1}L$ (middle), $D^{-1/2}LD^{-1/2}$ (right)",
        fontsize=16,
    )

    plt.tight_layout()
#-------------------------------------------------------------------
