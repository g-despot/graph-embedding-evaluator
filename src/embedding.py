import networkx as nx
import argparse
import os
from node2vec import Node2Vec
from gensim.models import Word2Vec
from src.algorithms.custom.deepwalk import DeepWalk as CustomDeepWalk
from src.algorithms.custom.node2vec import Node2Vec as CustomNode2Vec
from src.algorithms.node2vec.src import node2vec


def create_embedding(args: argparse.Namespace, G: nx.classes.graph.Graph) -> None:
    """Load the graph from a file in the input directory.

    Args:
        args (argparse.Namespace): The provided application arguments.
        G (networkx.classes.graph.Graph): The NetworkX graph object.
    """

    if args.method == 'node2vec_snap':
        """Implementation: https://github.com/aditya-grover/node2vec (SNAP - Stanford Network Analysis Project)."""

        G = node2vec.Graph(G, args.directed, args.p, args.q)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(args.num_walks, args.walk_length)
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(walks,
                         size=args.dimensions,
                         window=args.window_size,
                         seed=args.seed,
                         workers=args.workers,
                         iter=args.iter)
        model.wv.save_word2vec_format(args.output)

    elif args.method == 'node2vec_eliorc':
        """Implementation: https://github.com/eliorc/node2vec."""

        node2vecTmp = Node2Vec(graph=G,
                               walk_length=args.walk_length,
                               num_walks=args.num_walks,
                               dimensions=args.dimensions,
                               workers=args.workers)
        model = node2vecTmp.fit(window=args.window_size,
                                seed=args.seed,
                                workers=args.workers,
                                iter=args.iter)
        model.wv.save_word2vec_format(args.output)

    elif args.method == 'node2vec_custom':
        """Custom implementation."""

        model = CustomNode2Vec(num_walks=args.num_walks,
                               walk_length=args.walk_length,
                               p=args.p,
                               q=args.q,
                               size=args.dimensions,
                               window=args.window_size,
                               seed=args.seed,
                               workers=args.workers,
                               iter=args.iter)
        model.fit(G=G)
        model.save_embedding(args.output)

    elif args.method == 'deepwalk_phanein':
        """Implementation: https://github.com/phanein/deepwalk."""

        os.system(f"deepwalk --format edgelist --input {args.input} " +
                  f"--number-walks {args.num_walks} --representation-size {args.dimensions} " +
                  f"--walk-length {args.walk_length} --window-size {args.window_size} " +
                  f"--workers {args.workers} --seed {args.seed} --output {args.output}")

    elif args.method == 'deepwalk_custom':
        """Custom implementation"""

        model = CustomDeepWalk(num_walks=args.num_walks,
                               walk_length=args.walk_length,
                               size=args.dimensions,
                               window=args.window_size,
                               seed=args.seed,
                               workers=args.workers,
                               iter=args.iter)
        model.fit(G=G)
        model.save_embedding(args.output)

    else:
        raise ValueError(f'Invalid embedding algorithm: {args.method}')
