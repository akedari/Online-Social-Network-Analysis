"""
cluster.py
"""
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import json

listofcommunities = []

def intersect(a, b):
    return len(set(a).intersection(b))

def union(a, b):
    return  len(set(a).union(b))

def jaccardsim(user1, user2):

    user1_friends = user1[1]
    user2_friends = user2[1]

    numerator = intersect(user1_friends,user2_friends)
    denominator = union(user1_friends,user2_friends)

    if denominator == 0:
        return 0
    else:
        similarity = numerator/denominator
        return similarity

def create_graph(users):
    G = nx.Graph()

    nodes = []
    edges = []
    sim =0
    for user in users:
        nodes.append(user[0])

    n = len(users)

    for i in range(n-1):
        for j in range(i+1,n):
            sim = jaccardsim(users[i],users[j])
            # print('similarity between i=%d and j=%d  ------> %f'% (i, j, sim))
            if sim > 0.001:
                edge_tup1 = (nodes[i],nodes[j])
                edges.append(edge_tup1)

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G, n

    """totalnumberofusers = 0
    for user in users:
        G.add_node(user[0])
        friends = user[1]
        totalnumberofusers = totalnumberofusers + 1
        if (len(friends) > 0):
            for friend in friends:
                totalnumberofusers = totalnumberofusers + 1
                G.add_edge(user[0], friend)
    return G, totalnumberofusers"""


def get_subgraph(graph, min_degree):
    """Return a subgraph containing nodes whose degree is
    greater than or equal to min_degree.
    We'll use this in the main method to prune the original graph.

    Params:
      graph........a networkx graph
      min_degree...degree threshold
    Returns:
      a networkx graph, filtered as defined above.

    >>> subgraph = get_subgraph(example_graph(), 3)
    >>> sorted(subgraph.nodes())
    ['B', 'D', 'F']
    >>> len(subgraph.edges())
    2
    """
    ###TODO
    list_to_add_nodes = list()
    for node in graph:
        if graph.degree(node)>=min_degree:
            list_to_add_nodes.append(node)

    subgraph = graph.subgraph(list_to_add_nodes);
    return subgraph


def draw_network(graph, users, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).

    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.

    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """
    ###TODO
    G = nx.Graph(graph)
    label = {}
    fig = plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    """for user in users:
        label[user['screen_name']]=str(user['screen_name'])"""
    nx.draw_networkx(G, pos=pos, edge_color='gray', with_labels=False, node_size=50)
    nx.draw_networkx_labels(G, pos, label)
    plt.axis('off')
    plt.savefig(filename)
    # plt.show()
    # plt.close()


def girvan_newman(G, depth=0):
    """ Recursive implementation of the girvan_newman algorithm.
    See http://www-rohan.sdsu.edu/~gawron/python_for_ss/course_core/book_draft/Social_Networks/Networkx.html

    Args:
    G.....a networkx graph

    Returns:
    A list of all discovered communities,
    a list of lists of nodes. """

    if G.order() > 3 and G.order() < 70:
        listofcommunities.append(G.nodes())
        return

    if len(G.nodes()) <= 1:
        return

    def find_best_edge(G0):
        eb = nx.edge_betweenness_centrality(G0)
        return sorted(eb.items(), key=lambda x: x[1], reverse=True)[0][0]

    components = [c for c in nx.connected_component_subgraphs(G)]
    indent = '   ' * depth  # for printing
    while len(components) == 1:
        edge_to_remove = find_best_edge(G)
        G.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(G)]

    result = [c.nodes() for c in components]
    for c in components:
        girvan_newman(c, depth + 1)

    return result


def main():
    filename = 'data/followers.txt'
    users = []
    with open(filename, 'r') as f:
        for line in f:
            item = json.loads(line)
            users.append(item)

    output = []
    output.append('Number of users collected:')
    output.append(len(users))

    graph, totalnumberofusers = create_graph(users)

    subgraph = get_subgraph(graph,2)
    draw_network(subgraph, users, 'network.png')
    girvan_newman(subgraph)

    output.append('\nNumber of communities discovered:')
    output.append(len(listofcommunities))

    output.append('\nAverage number of users per community:')
    if len(listofcommunities) == 0:
        output.append(0)
    else:
        output.append(totalnumberofusers / len(listofcommunities))

    with open('temp/clustersdata.txt', 'w') as f:
        for entry in output:
            f.write(str(entry))
            f.write('\n')


if __name__ == '__main__':
    main()