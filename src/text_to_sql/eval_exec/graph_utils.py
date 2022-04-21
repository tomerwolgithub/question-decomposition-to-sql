import networkx as nx


def has_path(graph, start, end, path=[]):
    G = nx.Graph(graph)
    try:
        return nx.has_path(G, start, end)
    except:
        return None
    return None


def find_shortest_paths(graph, start, end):
    G = nx.Graph(graph)
    try:
        return [p for p in nx.all_shortest_paths(G, source=start, target=end)]
    except:
        None
    return None

# a sample graph
# graph = {'A': ['B', 'C', 'E'],
#              'B': ['A', 'C', 'D'],
#              'C': ['A', 'B', 'D', 'F'],
#              'D': ['A', 'B', 'C', 'E'],
#              'E': ['A', 'D', 'F'],
#              'F': ['C', 'E']}


# print(has_path(graph,"A","D"))
# print(find_shortest_paths(graph,"A","F"))
