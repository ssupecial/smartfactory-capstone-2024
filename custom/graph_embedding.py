import numpy as np
import networkx as nx
from typing import List, Optional
from karateclub.estimator import Estimator
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import numpy as np
import networkx as nx

import hashlib
import networkx as nx
from tqdm.auto import tqdm
from typing import List, Dict, Optional
from nodevectors import Node2Vec as FastNode2Vec
from node2vec import Node2Vec


class WeisfeilerLehmanHashing(object):  # Used for Graph2Vec
    """
    Weisfeiler-Lehman feature extractor class.

    Args:
        graph (NetworkX graph): NetworkX graph for which we do WL hashing.
        wl_iterations (int): Number of WL iterations.
        use_node_attribute (Optional[str]): Optional attribute name to be used.
        erase_base_feature (bool): Deleting the base features.
    """

    def __init__(
        self,
        graph: nx.classes.graph.Graph,
        wl_iterations: int,
        use_node_attribute: Optional[str],
        erase_base_features: bool,
    ):
        """
        Initialization method which also executes feature extraction.
        """
        self.wl_iterations = wl_iterations
        self.graph = graph
        self.use_node_attribute = use_node_attribute
        self.erase_base_features = erase_base_features
        self._set_features()
        self._do_recursions()

    def _set_features(self):
        """
        Creating the features.
        """
        if self.use_node_attribute is not None:
            # We retrieve the features of the nodes with the attribute name
            # `feature` and assign them into a dictionary with structure:
            # {node_a_name: feature_of_node_a}
            # Nodes without this feature will not appear in the dictionary.
            features = nx.get_node_attributes(self.graph, self.use_node_attribute)

            # We check whether all nodes have the requested feature
            if len(features) != self.graph.number_of_nodes():
                missing_nodes = []
                # We find up to five missing nodes so to make
                # a more informative error message.
                for node in tqdm(
                    self.graph.nodes,
                    total=self.graph.number_of_nodes(),
                    leave=False,
                    dynamic_ncols=True,
                    desc="Searching for missing nodes",
                ):
                    if node not in features:
                        missing_nodes.append(node)
                    if len(missing_nodes) > 5:
                        break
                raise ValueError(
                    (
                        "We expected for ALL graph nodes to have a node "
                        "attribute name `{}` to be used as part of "
                        "the requested embedding algorithm, but only {} "
                        "out of {} nodes has the correct attribute. "
                        "Consider checking for typos and missing values, "
                        "and use some imputation technique as necessary. "
                        "Some of the nodes without the requested attribute "
                        "are: {}"
                    ).format(
                        self.use_node_attribute,
                        len(features),
                        self.graph.number_of_nodes(),
                        missing_nodes,
                    )
                )
            # If so, we assign the feature set.
            self.features = features
        else:
            self.features = {
                node: self.graph.degree(node) for node in self.graph.nodes()
            }
        self.extracted_features = {k: [str(v)] for k, v in self.features.items()}

    def _erase_base_features(self):
        """
        Erasing the base features
        """
        for k, v in self.extracted_features.items():
            del self.extracted_features[k][0]

    def _do_a_recursion(self):
        """
        The method does a single WL recursion.

        Return types:
            * **new_features** *(dict of strings)* - The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.graph.nodes():
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = {
            k: self.extracted_features[k] + [v] for k, v in new_features.items()
        }
        return new_features

    def _do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.wl_iterations):
            self.features = self._do_a_recursion()
        if self.erase_base_features:
            self._erase_base_features()

    def get_node_features(self) -> Dict[int, List[str]]:
        """
        Return the node level features.
        """
        return self.extracted_features

    def get_graph_features(self) -> List[str]:
        """
        Return the graph level features.
        """
        return [
            feature
            for node, features in self.extracted_features.items()
            for feature in features
        ]


class Graph2Vec(Estimator):  # Input of Similarity
    def __init__(
        self,
        wl_iterations: int = 2,
        use_node_attribute: Optional[str] = None,
        dimensions: int = 128,
        workers: int = 4,
        down_sampling: float = 0.0001,
        epochs: int = 10,
        learning_rate: float = 0.025,
        min_count: int = 5,
        seed: int = 42,
        erase_base_features: bool = False,
    ):
        self.wl_iterations = wl_iterations
        self.use_node_attribute = use_node_attribute
        self.dimensions = dimensions
        self.workers = workers
        self.down_sampling = down_sampling
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
        self.erase_base_features = erase_base_features

    def fit(self, graphs: List[nx.classes.graph.Graph]):
        self._set_seed()
        # graphs = self._check_graphs(graphs)
        documents = [
            WeisfeilerLehmanHashing(
                graph=graph,
                wl_iterations=self.wl_iterations,
                use_node_attribute=self.use_node_attribute,
                erase_base_features=self.erase_base_features,
            )
            for graph in graphs
        ]
        documents = [
            TaggedDocument(words=doc.get_graph_features(), tags=[str(i)])
            for i, doc in enumerate(documents)
        ]

        self.model = Doc2Vec(
            documents,
            vector_size=self.dimensions,
            window=0,
            min_count=self.min_count,
            dm=0,
            sample=self.down_sampling,
            workers=self.workers,
            epochs=self.epochs,
            alpha=self.learning_rate,
            seed=self.seed,
        )

        self._embedding = [self.model.dv[str(i)] for i, _ in enumerate(documents)]

    def get_embedding(self) -> np.array:
        return np.array(self._embedding)

    def infer(self, graphs) -> np.array:
        self._set_seed()
        # graphs = self._check_graphs(graphs)
        documents = [
            WeisfeilerLehmanHashing(
                graph=graph,
                wl_iterations=self.wl_iterations,
                use_node_attribute=self.use_node_attribute,
                erase_base_features=self.erase_base_features,
            )
            for graph in graphs
        ]

        documents = [doc.get_graph_features() for _, doc in enumerate(documents)]

        embedding = np.array(
            [
                self.model.infer_vector(
                    doc, alpha=self.learning_rate, min_alpha=0.00001, epochs=self.epochs
                )
                for doc in documents
            ]
        )

        return embedding


def node2vector(graph, type="faster", dimensions=20):  # Input of Transformer Model
    if type == "faster":
        node2vec = FastNode2Vec(
            n_components=dimensions,
            walklen=30,
            return_weight=1,
            neighbor_weight=1,
            epochs=50,
        )
        node2vec.fit(graph)
        node_embeddings = np.array(
            [node2vec.model.wv[str(node)] for node in graph.nodes()]
        )

    else:
        node2vec = Node2Vec(
            graph,
            dimensions=dimensions,
            walk_length=30,
            num_walks=50,
            workers=4,
            p=1,
            q=1,
        )
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        node_embeddings = np.array([model.wv[str(node)] for node in graph.nodes()])

    # EMBEDDING_FILENAME = "test.emb"
    # EMBEDDING_MODEL_FILENAME = "test.model"

    # # Save embeddings for later use
    # model.wv.save_word2vec_format(EMBEDDING_FILENAME)

    # # Save model for later use
    # model.save(EMBEDDING_MODEL_FILENAME)

    return node_embeddings
