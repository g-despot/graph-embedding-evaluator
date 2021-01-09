import logging
import networkx as nx


logger = logging.getLogger('evaluator')


def load_graph(is_weighted: bool, is_directed: bool, file_path: str) -> nx.classes.graph.Graph:
    """Load the graph from a file in the input directory.

    Args:
        is_weighted (boolean): Is the graph weighted.
        is_directed (boolean): Is the graph directed.
        file_path (str): Path to the file with the input graph.

    Returns:
        G (networkx.classes.graph.Graph): A NetworkX graph object.
    """

    if is_weighted:
        G = nx.read_edgelist(file_path, nodetype=int, data=(
            ('weight', float),), create_using=nx.Graph())
    else:
        G = nx.read_edgelist(file_path, nodetype=int, create_using=nx.Graph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if is_directed:
        G = G.to_undirected()

    return G


def print_graph_info(G: nx.classes.graph.Graph, graph_name: str) -> None:
    """Print information about the graph.

    Args:
        G (networkx.classes.graph.Graph): A NetworkX graph object.
        graph_name (str): The name of the graph.
    """

    number_of_nodes = nx.number_of_nodes(G)
    number_of_edges = nx.number_of_edges(G)
    density = nx.density(G)

    logger.info(f'\nInformation about the {graph_name}')
    logger.info(
        f'Number of nodes: {number_of_nodes}\tNumber of edges: {number_of_edges}\tDensity: {density}\n')


def load_embedding(file_path: str) -> dict:
    """Load the node embeddings from a file.

    Args:
        file_path (str): Path to the file with the node embeddings.

    Results:
        embedding_dict (dict): A dictionary of node embedding vectors with nodes as keys.
    """

    embedding_dict = {}
    with open(file_path) as f:
        for line in f:
            vector = [float(i) for i in line.strip().split()]
            embedding_dict[vector[0]] = vector[1:]
        f.close()

    return embedding_dict
