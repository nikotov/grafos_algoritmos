import argparse
import heapq
import json
from collections import defaultdict


def parse():
    """
    Parse the command line arguments and return the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Shortest paths from a root vertex (Dijkstra)")
    parser.add_argument("input_file", type=str, help="Path to the input file containing the graph")
    parser.add_argument("--output_file", type=str, default="mst_output.json",
                        help="Path to the output file for the result")
    parser.add_argument("--root", type=str, default=None,
                        help="Root vertex for the paths (defaults to the first vertex in V)")
    return parser.parse_args()


def construct_graph(input_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    vertices = data["grafo"]["V"]
    matrix = data["grafo"]["EC"]

    n = len(vertices)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("EC must be an n x n matrix matching V")

    # Edge list (u, v, weight); 0 means no edge. Upper triangle only (undirected).
    edges = [
        (vertices[i], vertices[j], matrix[i][j])
        for i in range(n)
        for j in range(i + 1, n)
        if matrix[i][j] != 0
    ]
    if any(w < 0 for _, _, w in edges):
        raise ValueError("Dijkstra does not support negative edge weights")
    return vertices, edges


def dijkstra(vertices, edges, root):
    """Return {vertex: [root, ..., vertex, total_cost]} for every reachable vertex except root."""
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    index = {v: i for i, v in enumerate(vertices)}  # tie-breaker so heap never compares labels
    dist = {root: 0}
    prev = {}
    heap = [(0, index[root], root)]
    done = set()
    while heap:
        d, _, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        for v, w in adj[u]:
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, index[v], v))

    result = {}
    for v in vertices:
        if v == root or v not in dist:
            continue
        path = [v]
        while path[-1] != root:
            path.append(prev[path[-1]])
        path.reverse()
        result[v] = path + [dist[v]]
    return result


def write_output(output_file, paths):
    # One entry per line, lists kept on a single line.
    items = [f"        {json.dumps(k)}: {json.dumps(v)}" for k, v in paths.items()]
    with open(output_file, "w") as f:
        f.write('{\n    "output": {\n')
        f.write(",\n".join(items))
        f.write("\n    }\n}\n")


def main():
    args = parse()
    vertices, edges = construct_graph(args.input_file)
    root = args.root if args.root is not None else vertices[0]
    if root not in vertices:
        raise ValueError(f"Root vertex {root!r} is not in V")
    paths = dijkstra(vertices, edges, root)
    write_output(args.output_file, paths)


if __name__ == "__main__":
    main()