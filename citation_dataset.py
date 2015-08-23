import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sci_sp

if __name__ == "__main__":
    net = nx.read_edgelist("/home/curuinor/data/Cit-HepPh.txt")
    net = net.subgraph(map(str, range(3000)))
    print "net created"
    net_arr = nx.to_numpy_matrix(net)
    net_arr = np.squeeze(np.asarray(net_arr))
    degrees = net_arr.sum(axis=0)
    degrees = np.sort(degrees)
    plt.plot(degrees)
    plt.title("citation network degrees")
    plt.savefig("./pics/citation_degrees")
