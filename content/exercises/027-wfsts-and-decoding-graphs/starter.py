"""Weighted finite-state transducers and a CTC topology.

Fill in the five functions, then press Run. Epsilon is the string "<eps>".
A transducer is a dict with "start", "arcs", and "finals". Each arc is
(src, dst, ilabel, olabel, weight). "finals" maps a state to its final cost.
"""

import math

import numpy as np

from speechlab import show_plot

EPS = "<eps>"


def compose(a, b):
    """The composition of transducers `a` and `b`.

    Pair an arc of `a` with an arc of `b` when `a`'s output equals `b`'s
    input and that label is not "<eps>". The composed arc keeps `a`'s input,
    `b`'s output, and the sum of the costs. An output "<eps>" on `a` moves
    only `a`. An input "<eps>" on `b` moves only `b`. A product state is
    final when both states are, and the final cost is the sum.

    Return a transducer of the same shape. State ids may be any ints.
    """
    raise NotImplementedError("compose")


def shortest_path(fst):
    """The accepting path of least total cost.

    Return `(cost, inputs, outputs)`. `inputs` and `outputs` are tuples of
    labels with "<eps>" removed, and `cost` includes the final weight. Costs
    are non-negative. If nothing is accepted, return `(math.inf, (), ())`.
    """
    raise NotImplementedError("shortest_path")


def n_best(fst, n):
    """The `n` cheapest accepting paths, as a list of `(cost, inputs, outputs)`.

    Sort by increasing cost. The graph is acyclic. Return every path when
    fewer than `n` exist.
    """
    raise NotImplementedError("n_best")


def total_weight(fst):
    """The log-semiring total of every accepting path, as a cost.

    Path costs are negative log probabilities. The total is
    `-log(sum(exp(-cost)))`, which is `-np.logaddexp.reduce(-costs)`.
    Return `math.inf` when the graph accepts nothing. The graph is acyclic.
    """
    raise NotImplementedError("total_weight")


def ctc_topology(target, blank=0):
    """The CTC graph for `target`.

    `target` is a sequence of ints, and none of them equals `blank`. Every
    arc has cost 0, and its input and output are the same symbol. The graph
    accepts a string of ints exactly when collapsing that string leaves
    `target`: merge each run of the same symbol, then drop `blank`. A
    repeated label needs a blank between its copies. Two different labels
    may follow each other with or without a blank.

    Return a transducer. The start state is not final unless `target` is empty.
    """
    raise NotImplementedError("ctc_topology")


if __name__ == "__main__":
    lexicon = {
        "start": 0,
        "arcs": [
            (0, 2, "ah", "a", 0.2),
            (0, 1, "k", EPS, 0.1),
            (1, 2, "t", "cat", 0.3),
        ],
        "finals": {2: 0.0},
    }
    grammar = {
        "start": 0,
        "arcs": [
            (0, 1, "a", "a", 0.5),
            (0, 2, "cat", "cat", 0.8),
            (1, 2, "cat", "cat", 0.3),
        ],
        "finals": {1: 0.4, 2: 0.2},
    }
    graph = compose(lexicon, grammar)
    for cost, inputs, outputs in n_best(graph, 5):
        print(" ".join(inputs), "->", " ".join(outputs), f"cost {cost:.1f}")
    print("log-semiring total", total_weight(graph))
    show_plot(
        np.array([cost for cost, _, _ in n_best(graph, 5)]),
        label="Tropical cost of each path in the composed lexicon and bigram",
        xlabel="Path index, cheapest first",
        ylabel="Cost",
    )
