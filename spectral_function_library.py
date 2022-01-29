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
