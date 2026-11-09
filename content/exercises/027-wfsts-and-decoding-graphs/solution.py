"""Weighted transducers: composition, shortest paths, and a CTC topology."""

import heapq
import math
from collections import defaultdict

import numpy as np

from speechlab import show_plot

EPS = "<eps>"


def _outgoing(fst):
    """Arcs grouped by source state: dst, input, output, weight."""
    grouped = defaultdict(list)
    for src, dst, ilabel, olabel, weight in fst["arcs"]:
        grouped[src].append((dst, ilabel, olabel, float(weight)))
    return grouped


def compose(a, b):
    """Compose `a` with `b`. Costs add. Epsilon is the string '<eps>'.

    A real output of `a` pairs with the same input of `b`. An output epsilon
    on `a` moves only `a`. An input epsilon on `b` moves only `b`. A product
    state is final when both states are final, and those costs add too.
    """
    arcs_a = _outgoing(a)
    arcs_b = _outgoing(b)
    start = (a["start"], b["start"])
    index = {start: 0}
    arcs = []
    queue = [start]
    head = 0
    while head < len(queue):
        qa, qb = queue[head]
        head += 1
        src = index[(qa, qb)]
        for da, ia, oa, wa in arcs_a.get(qa, []):
            if oa == EPS:
                arcs.append((src, _pid(index, queue, (da, qb)), ia, EPS, wa))
                continue
            for db, ib, ob, wb in arcs_b.get(qb, []):
                if oa == ib:
                    dst = _pid(index, queue, (da, db))
                    arcs.append((src, dst, ia, ob, wa + wb))
        for db, ib, ob, wb in arcs_b.get(qb, []):
            if ib == EPS:
                arcs.append((src, _pid(index, queue, (qa, db)), EPS, ob, wb))
    finals = {}
    for (qa, qb), state in index.items():
        if qa in a["finals"] and qb in b["finals"]:
            finals[state] = float(a["finals"][qa]) + float(b["finals"][qb])
    return {"start": 0, "arcs": arcs, "finals": finals}


def _pid(index, queue, pair):
    """The integer id of a product state, enqueueing it on first sight."""
    state = index.get(pair)
    if state is None:
        state = len(index)
        index[pair] = state
        queue.append(pair)
    return state


def _paths(fst):
    """Every accepting path of an acyclic graph: (cost, inputs, outputs).

    Inputs and outputs omit '<eps>'. The cost includes the final weight.
    A final state with outgoing arcs contributes both the path that stops
    there and the paths that continue.
    """
    outgoing = _outgoing(fst)
    found = []

    def walk(state, cost, inputs, outputs, seen):
        if state in seen:
            return
        if state in fst["finals"]:
            found.append(
                (cost + float(fst["finals"][state]), inputs, outputs)
            )
        seen = seen | {state}
        for dst, ilabel, olabel, weight in outgoing.get(state, []):
            nin = inputs if ilabel == EPS else inputs + (ilabel,)
            nout = outputs if olabel == EPS else outputs + (olabel,)
            walk(dst, cost + weight, nin, nout, seen)

    walk(fst["start"], 0.0, (), (), frozenset())
    return found


def shortest_path(fst):
    """The accepting path of least cost, as (cost, inputs, outputs).

    Costs are non-negative. The search is Dijkstra, so a cycle cannot make
    it run forever, and the first time a state is settled its cost is final.
    With no accepting path, return `(math.inf, (), ())`.
    """
    outgoing = _outgoing(fst)
    start = fst["start"]
    dist = {start: 0.0}
    parent = {start: None}
    heap = [(0.0, 0, start)]
    tie = 0
    while heap:
        cost, _, state = heapq.heappop(heap)
        if cost > dist[state] + 1e-12:
            continue
        for dst, ilabel, olabel, weight in outgoing.get(state, []):
            nxt = cost + weight
            if nxt + 1e-12 < dist.get(dst, math.inf):
                dist[dst] = nxt
                parent[dst] = (state, ilabel, olabel)
                tie += 1
                heapq.heappush(heap, (nxt, tie, dst))
    best = None
    for state, final in fst["finals"].items():
        if state not in dist:
            continue
        total = dist[state] + float(final)
        inputs, outputs = _labels(parent, state)
        if best is None or total < best[0]:
            best = (total, inputs, outputs)
    if best is None:
        return (math.inf, (), ())
    return best


def _labels(parent, state):
    """Input and output symbols along the settled path, epsilons removed."""
    chain = []
    while parent[state] is not None:
        state, ilabel, olabel = parent[state]
        chain.append((ilabel, olabel))
    inputs, outputs = [], []
    for ilabel, olabel in reversed(chain):
        if ilabel != EPS:
            inputs.append(ilabel)
        if olabel != EPS:
            outputs.append(olabel)
    return tuple(inputs), tuple(outputs)


def n_best(fst, n):
    """The `n` accepting paths of least cost, cheapest first.

    The graph must be acyclic. `n` larger than the number of paths returns
    every path.
    """
    paths = _paths(fst)
    paths.sort(key=lambda item: item[0])
    return paths[: int(n)]


def total_weight(fst):
    """Log-semiring total of every accepting path, as a cost.

    Weights are negative log probabilities, so the total is
    `-log(sum(exp(-cost)))`. An empty graph returns `math.inf`. The graph
    must be acyclic, because a cycle would make the sum infinite.
    """
    costs = [cost for cost, _, _ in _paths(fst)]
    if not costs:
        return math.inf
    return float(-np.logaddexp.reduce(-np.array(costs, dtype=float)))


def ctc_topology(target, blank=0):
    """The CTC graph for `target`: blank self-loops, label self-loops, and a
    blank between two copies of the same label.

    Every arc has cost 0, and its input and output are the same symbol. A
    string is in the graph when collapsing runs and then dropping `blank`
    leaves `target`. Different neighbors may skip the blank between them.
    """
    labels = [int(symbol) for symbol in target]
    blank = int(blank)
    if any(symbol == blank for symbol in labels):
        raise ValueError("target labels must not include the blank")
    if not labels:
        return {
            "start": 0,
            "arcs": [(0, 0, blank, blank, 0.0)],
            "finals": {0: 0.0},
        }
    arcs = [(0, 0, blank, blank, 0.0)]
    for index, label in enumerate(labels):
        label_state = 2 * index + 1
        blank_state = 2 * index + 2
        enter_from = 0 if index == 0 else 2 * (index - 1) + 2
        arcs.append((enter_from, label_state, label, label, 0.0))
        if index > 0 and labels[index - 1] != label:
            arcs.append((2 * (index - 1) + 1, label_state, label, label, 0.0))
        arcs.append((label_state, label_state, label, label, 0.0))
        arcs.append((label_state, blank_state, blank, blank, 0.0))
        arcs.append((blank_state, blank_state, blank, blank, 0.0))
    last = len(labels) - 1
    return {
        "start": 0,
        "arcs": arcs,
        "finals": {2 * last + 1: 0.0, 2 * last + 2: 0.0},
    }


def _lexicon():
    """Phones to words: ah writes a, and k t writes cat, with k silent."""
    return {
        "start": 0,
        "arcs": [
            (0, 2, "ah", "a", 0.2),
            (0, 1, "k", EPS, 0.1),
            (1, 2, "t", "cat", 0.3),
        ],
        "finals": {2: 0.0},
    }


def _grammar():
    """A toy bigram over a and cat. Costs are negative log probabilities."""
    return {
        "start": 0,
        "arcs": [
            (0, 1, "a", "a", 0.5),
            (0, 2, "cat", "cat", 0.8),
            (1, 2, "cat", "cat", 0.3),
        ],
        "finals": {1: 0.4, 2: 0.2},
    }


if __name__ == "__main__":
    graph = compose(_lexicon(), _grammar())
    paths = n_best(graph, 5)
    for cost, inputs, outputs in paths:
        phones = " ".join(inputs)
        words = " ".join(outputs)
        print(f"{phones} -> {words}    cost {cost:.1f}")
    print(f"shortest path cost {shortest_path(graph)[0]:.1f}")
    print(f"log-semiring total {total_weight(graph):.4f}")
    show_plot(
        np.array([cost for cost, _, _ in paths]),
        label="Tropical cost of each path in the composed lexicon and bigram",
        xlabel="Path index, cheapest first",
        ylabel="Cost",
    )
    topo = ctc_topology([1, 1], blank=0)
    print(f"CTC topology for [1, 1] has {len(topo['arcs'])} arcs")
