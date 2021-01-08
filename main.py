import argparse
import logging
import networkx as nx
import src.embedding as embedding
import src.evaluation as evaluation
import src.utils as utils
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from stellargraph.data import EdgeSplitter
from sklearn.model_selection import train_test_split


logger = logging.getLogger('evaluator')
logging.getLogger().addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

_here = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    """Parse the application arguments.

    Returns:
        args (argparse.Namespace): Application arguments.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--input', required=True,
                        help='Input graph in edge list format in the /input directory.')
    parser.add_argument('--output',
                        help='File name to save the graph embeddings in the /embeddings directory.', required=True)
    parser.add_argument('--results',
                        help='File name to save the evaluation results in the /results directory.')
    parser.add_argument('--dimensions', default=128, type=int,
                        help='Dimensionality of the word vectors. (default: 128)')
    parser.add_argument('--walk-length', default=64, type=int,
                        help='The number of nodes in each walk. (default: 64)')
    parser.add_argument('--num-walks', default=32, type=int,
                        help='Number of walks from each node. (default: 32)')
    parser.add_argument('--p', default=2.0, type=float,
                        help='Node2vec return parameter p. (default: 2)')
    parser.add_argument('--q', default=1.0, type=float,
                        help='Node2vec in-out parameter q. (default: 1)')
    parser.add_argument('--workers', default=1, type=int,
                        help='Number of worker threads to train the model. (default: 1)')
    parser.add_argument('--seed', default=0, type=int,
                        help='A seed for the random number generator. (default: 0)')
    parser.add_argument('--test-percentage', default=0.1, type=float,
                        help='Percentage of graph edges that should be used for testing classifiers.')
    parser.add_argument('--train-percentage', default=0.1, type=float,
                        help='Percentage of graph edges that should be used for training classifiers.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Maximum distance between the current and predicted word within a sentence. (default: 10)')
    parser.add_argument('--weighted', type=bool, default=False,
                        help='Denotes if the graph is weighted. (default: False)')
    parser.add_argument('--directed', type=bool, default=False,
                        help='Denotes if the graph is directed. (default: False)')
    parser.add_argument('--iter', default=1, type=int,
                        help='Number of iterations (epochs) over the corpus.')
    parser.add_argument('--method', required=True, choices=['node2vec_snap',
                                                            'node2vec_eliorc',
                                                            'node2vec_custom',
                                                            'deepwalk_phanein',
                                                            'deepwalk_custom'], help='The graph embedding algorithm and specific implementation.')
    parser.add_argument('--classifier', required=True, choices=['logisticalregression',
                                                                'randomforest',
                                                                'gradientboost', ], help='The classifier for link prediction evaluation.')
    args = parser.parse_args()

    args.dataset = args.input
    args.input = _here.parent.joinpath("input/" + args.input)
    args.output = _here.parent.joinpath("embeddings/" + args.output)
    if args.results:
        args.results = _here.parent.joinpath("results/" + args.results)

    return args


def main():
    """Load the graph, create the embeddings, evaluate them with link prediction and save the results."""

    args = parse_args()

    graph = utils.load_graph(args.weighted, args.directed, args.input)
    utils.print_graph_info(graph, "original graph")

    graph.remove_nodes_from(list(nx.isolates(graph)))
    utils.print_graph_info(graph, "graph without isolates")

    edge_splitter_test = EdgeSplitter(graph)

    graph_test, X_test_edges, y_test = edge_splitter_test.train_test_split(
        p=args.test_percentage, method="global"
    )

    edge_splitter_train = EdgeSplitter(graph_test, graph)
    graph_train, X_edges, y = edge_splitter_train.train_test_split(
        p=args.train_percentage, method="global"
    )
    X_train_edges, X_model_selection_edges, y_train, y_model_selection = train_test_split(
        X_edges, y, train_size=0.75, test_size=0.25)

    logger.info(f'\nEmbedding algorithm started.')
    start = time.time()

    embedding.create_embedding(args, graph_train)
    time_diff = time.time() - start
    logger.info(f'\nEmbedding algorithm finished in {time_diff:.2f} seconds.')

    embeddings = utils.load_embedding(args.output)

    logger.info(f'\nEmbedding evaluation started.')
    start = time.time()
    results = evaluation.evaluate(args.classifier,
                                  embeddings,
                                  X_train_edges,
                                  y_train,
                                  X_model_selection_edges,
                                  y_model_selection)

    time_diff = time.time() - start
    logger.info(f'Embedding evaluation finished in {time_diff:.2f} seconds.')

    best_result = max(results, key=lambda result: result["roc_auc"])

    logger.info(
        f"\nBest roc_auc_score on train set using '{best_result['binary_operator'].__name__}': {best_result['roc_auc']}.")

    logger.info(f'\nEmbedding algorithm started.')
    start = time.time()

    embedding.create_embedding(args, graph_test)
    time_diff = time.time() - start
    logger.info(f'\nEmbedding algorithm finished in {time_diff:.2f} seconds.')

    embedding_test = utils.load_embedding(args.output)

    roc_auc, average_precision, accuracy, f1 = evaluation.evaluate_model(best_result["classifier"],
                                                                         embedding_test,
                                                                         best_result["binary_operator"],
                                                                         X_test_edges,
                                                                         y_test)

    logger.info(
        f"Scores on test set using '{best_result['binary_operator'].__name__}'.")
    logger.info(f"roc_auc_score: {roc_auc}")
    logger.info(f"average_precision_score: {average_precision}")
    logger.info(f"accuracy_score: {accuracy}")
    logger.info(f"f1_score on test set using: {f1}\n")

    if(args.results):
        evaluation.save_evaluation_results(args.dataset,
                                           args.method,
                                           args.classifier,
                                           (roc_auc, average_precision, accuracy, f1),
                                           args.results)


if __name__ == "__main__":
    """Main entry point of the application."""

    main()
