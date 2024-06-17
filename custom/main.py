import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time

from utils import generate_instance, print_instance, make_graph
from similarity import similarity_graph2vec
from graph_embedding import node2vector
from node2vec import Node2Vec


def test_similarity():
    num_jobs = 100  # number of jobs
    num_machines = 20  # number of machines

    # Generate instance
    processing_time_matrix, machine_matrix = generate_instance(
        num_jobs, num_machines, random_seed=10
    )
    # embeddings = make_vectors(processing_time_matrix, machine_matrix, 20)

    G = make_graph(processing_time_matrix, machine_matrix)

    processing_time_matrix2, machine_matrix2 = generate_instance(
        num_jobs, num_machines, random_seed=100
    )
    G2 = make_graph(processing_time_matrix2, machine_matrix2)

    similarity = similarity_graph2vec(G, G2)
    print(similarity)


def test_random_mask():
    # deprecated_time_matrix, deprecated_machine_matrix = random_mask(
    #     processing_time_matrix, machine_matrix, num_jobs, num_machines, 8, 8
    # )
    # deprecated_graph = make_graph(deprecated_time_matrix, deprecated_machine_matrix)
    pass


def make_vectors(processing_time_matrix, machine_matrix, dimensions=20):
    G = make_graph(processing_time_matrix, machine_matrix)
    node2vec = Node2Vec(
        G, dimensions=dimensions, walk_length=30, num_walks=200, workers=4, p=1, q=1
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    node_embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
    return node_embeddings


def test_embedding():
    num_jobs = 100  # number of jobs
    num_machines = 20  # number of machines

    # Generate instance
    processing_time_matrix, machine_matrix = generate_instance(
        num_jobs, num_machines, random_seed=10
    )
    # embeddings = make_vectors(processing_time_matrix, machine_matrix, 20)

    G = make_graph(processing_time_matrix, machine_matrix)

    cur = time.time()
    node_embeddings = node2vector(G, type="faster")
    print("Time:", time.time() - cur)
    # print(node_embeddings)
    print("Shape")
    print(node_embeddings.shape)


if __name__ == "__main__":
    test_embedding()
