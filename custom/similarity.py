import networkx as nx
from grakel import Graph
from grakel.kernels import WeisfeilerLehman
import time
import numpy as np
from graph_embedding import WeisfeilerLehmanHashing, Graph2Vec


def edge_similarity(graph1, graph2):
    cur = time.time()
    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())

    common_edges = edges1.intersection(edges2)
    all_edges = edges1.union(edges2)

    if len(edges1) == 0:
        return 0.0

    similarity = len(common_edges) / len(all_edges)
    fin = time.time()
    return similarity, fin - cur


def edge_similarity_grakel(
    source_graph: nx.DiGraph, target_graph: nx.DiGraph, n_iter: int
):
    cur = time.time()

    def nx_to_grakel(G):
        node_labels = nx.get_node_attributes(G, "machine")  # node attribute
        edge_labels = nx.get_edge_attributes(G, "type")  # edge attribute
        return Graph(list(G.edges()), node_labels=node_labels, edge_labels=edge_labels)

    wl_kernel = WeisfeilerLehman(n_iter=n_iter)
    K = wl_kernel.fit_transform(
        [nx_to_grakel(source_graph), nx_to_grakel(target_graph)]
    )

    fin = time.time()

    similarity = K[0, 1]  # Source-Target graph similarity
    ratio = K[0, 0]  # Source graph similarity
    return similarity / ratio, fin - cur


def edge_similarity_cosine(source_graph: nx.DiGraph, target_graph: nx.DiGraph):
    pass


def edge_similarity_mcs(source_graph: nx.DiGraph, target_graph: nx.DiGraph):
    # Maximum Common Subgraph
    cur = time.time()
    mcs = nx.algorithms.isomorphism.GraphMatcher(source_graph, target_graph)
    max_common_subgraph = max(
        (len(subgraph), subgraph) for subgraph in mcs.subgraph_isomorphisms_iter()
    )[1]
    fin = time.time()

    print("Maximum Common Subgraph:", max_common_subgraph)
    return len(max_common_subgraph) / len(source_graph.edges()), fin - cur


def edge_similarity_sc(source_graph: nx.DiGraph, target_graph: nx.DiGraph):
    # Spectral Comparison
    cur = time.time()
    A1 = nx.adjacency_matrix(source_graph).todense()
    A2 = nx.adjacency_matrix(target_graph).todense()

    # 고유값 계산
    eigenvalues1 = np.linalg.eigvals(A1)
    eigenvalues2 = np.linalg.eigvals(A2)

    # 유클리드 거리 계산
    euclidean_distance = np.linalg.norm(eigenvalues1 - eigenvalues2)
    # print("Euclidean Distance:", euclidean_distance)

    # 코사인 유사도 계산
    cosine_similarity = np.dot(eigenvalues1, eigenvalues2) / (
        np.linalg.norm(eigenvalues1) * np.linalg.norm(eigenvalues2)
    )
    fin = time.time()

    # return cosine_similarity, fin - cur
    return cosine_similarity


def similarity_graph2vec(graph1, graph2):
    model = Graph2Vec(use_node_attribute="machine", dimensions=1000, wl_iterations=5)
    model.fit([graph1, graph2])
    embedding1, embedding2 = model.get_embedding()

    cosine_similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )

    euclidean_distance = np.linalg.norm(embedding1 - embedding2)

    return cosine_similarity
