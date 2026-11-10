"""Checks for composition, semiring path scores, and the CTC topology.

Two wrong implementations are named here so a reader can see what fails.

`compose_drop_eps` deletes every arc whose input or output is '<eps>'.
The lexicon path that writes 'cat' is only reachable through the epsilon
arc on k, so that compose returns no cat path.

`shortest_path_add_probabilities` adds the arc weights and keeps the
largest sum, as if the weights were probabilities to accumulate. On the
two-path graph below that sum is 0.9. The tropical path is the minimum
sum, 0.4.

`context_free_c` reads every triphone from one looping state and writes
its middle phone. It accepts `sil-k+t` followed by `ah-t+sil`, whose
windows do not overlap, and a string that starts with `ah-k+t`, which
has no silence at the boundary. `hclg_reversed` composes G, L, C, and H
in the opposite order, so no output label meets an input label and the
graph has no path.

`ctc_topology_skip_repeat_blank` draws an arc from one label straight to
the same label again. The string (1, 1) is then accepted for target
[1, 1], but collapse((1, 1)) is [1].
"""

import importlib.util
import os
from pathlib import Path

import numpy as np

from speechcheck import check, fail, has

EPS = "<eps>"


def _outgoing(fst):
    grouped = {}
    for src, dst, ilabel, olabel, weight in fst["arcs"]:
        grouped.setdefault(src, []).append((dst, ilabel, olabel, float(weight)))
    return grouped


def _paths(fst, limit=12):
    """Accepting paths as (cost, inputs, outputs), epsilons removed.

    A state already on the current path is not entered again, so a cycle
    cannot hang the check. The lesson's graphs are acyclic.
    """
    outgoing = _outgoing(fst)
    found = []

    def walk(state, cost, inputs, outputs, seen, depth):
        if depth > limit or state in seen:
            return
        if state in fst["finals"]:
            found.append((cost + float(fst["finals"][state]), inputs, outputs))
        seen = seen | {state}
        for dst, ilabel, olabel, weight in outgoing.get(state, []):
            nin = inputs if ilabel == EPS else inputs + (ilabel,)
            nout = outputs if olabel == EPS else outputs + (olabel,)
            walk(dst, cost + weight, nin, nout, seen, depth + 1)

    walk(fst["start"], 0.0, (), (), frozenset(), 0)
    return found


def _aligns(out_a, in_b):
    """True when some one-sided epsilon alignment matches the two tapes."""

    def rec(i, j, depth):
        if depth > 24:
            return False
        if i == len(out_a) and j == len(in_b):
            return True
        if i < len(out_a) and out_a[i] == EPS and rec(i + 1, j, depth + 1):
            return True
        if j < len(in_b) and in_b[j] == EPS and rec(i, j + 1, depth + 1):
            return True
        if (
            i < len(out_a)
            and j < len(in_b)
            and out_a[i] == in_b[j]
            and out_a[i] != EPS
            and rec(i + 1, j + 1, depth + 1)
        ):
            return True
        return False

    return rec(0, 0, 0)


def _pair_paths(a, b):
    """One result per accepting path pair whose output and input align.

    Epsilon steps are taken on one tape at a time. Each pair is counted
    once, which matches composition when epsilons sit on only one tape.
    """
    results = []
    for inputs_a, outputs_a, weight_a in _paths_kept(a):
        for inputs_b, outputs_b, weight_b in _paths_kept(b):
            if _aligns(outputs_a, inputs_b):
                results.append(
                    (
                        tuple(symbol for symbol in inputs_a if symbol != EPS),
                        tuple(symbol for symbol in outputs_b if symbol != EPS),
                        weight_a + weight_b,
                    )
                )
    return results


def _paths_kept(fst):
    """Accepting paths that still contain epsilon symbols, for alignment."""
    outgoing = _outgoing(fst)
    found = []

    def walk(state, cost, inputs, outputs, seen, depth):
        if depth > 12 or state in seen:
            return
        if state in fst["finals"]:
            total = cost + float(fst["finals"][state])
            found.append((inputs, outputs, total))
        seen = seen | {state}
        for dst, ilabel, olabel, weight in outgoing.get(state, []):
            walk(
                dst,
                cost + weight,
                inputs + (ilabel,),
                outputs + (olabel,),
                seen,
                depth + 1,
            )

    walk(fst["start"], 0.0, (), (), frozenset(), 0)
    return found


def _sorted(paths):
    return sorted(
        (tuple(inputs), tuple(outputs), round(float(weight), 6))
        for weight, inputs, outputs in paths
    )


def _same(got_fst, want, label):
    """Compare composed paths to `(inputs, outputs, weight)` triples."""
    got = _sorted(_paths(got_fst))
    want = sorted(
        (tuple(inputs), tuple(outputs), round(float(weight), 6))
        for inputs, outputs, weight in want
    )
    if got != want:
        fail(f"{label}: expected paths {want}, got {got}")


def _fst(arcs, finals, start=0):
    return {"start": start, "arcs": arcs, "finals": finals}


def _lexicon():
    return _fst(
        [
            (0, 2, "ah", "a", 0.2),
            (0, 1, "k", EPS, 0.1),
            (1, 2, "t", "cat", 0.3),
        ],
        {2: 0.0},
    )


def _grammar():
    return _fst(
        [
            (0, 1, "a", "a", 0.5),
            (0, 2, "cat", "cat", 0.8),
            (1, 2, "cat", "cat", 0.3),
        ],
        {1: 0.4, 2: 0.2},
    )


@check("compose matches every path pair of an epsilon-free transducer")
def test_compose_epsilon_free(sub):
    compose = has(sub, "compose")
    first = _fst([(0, 1, "x", "p", 0.3), (0, 1, "y", "q", 0.5)], {1: 0.0})
    second = _fst([(0, 1, "p", "P", 0.4), (0, 1, "q", "Q", 0.6)], {1: 0.1})
    _same(compose(first, second), _pair_paths(first, second), "epsilon-free pair")


@check("compose keeps a path that is only reachable through an epsilon arc")
def test_compose_epsilon(sub):
    # compose_drop_eps removes the k:<eps> arc and then cannot write cat.
    compose = has(sub, "compose")
    lexicon, grammar = _lexicon(), _grammar()
    _same(compose(lexicon, grammar), _pair_paths(lexicon, grammar), "lexicon with G")
    got = _sorted(_paths(compose(lexicon, grammar)))
    want = sorted(
        [
            (("ah",), ("a",), round(1.1, 6)),
            (("k", "t"), ("cat",), round(1.4, 6)),
        ]
    )
    if got != want:
        fail(f"lexicon composed with G: expected {want}, got {got}")

    # An input epsilon on the second transducer has to move too.
    left = _fst([(0, 1, "a", "x", 0.4)], {1: 0.0})
    right = _fst([(0, 1, EPS, "s", 0.1), (1, 2, "x", "y", 0.2)], {2: 0.0})
    _same(compose(left, right), _pair_paths(left, right), "input epsilon on B")


@check("compose matches enumeration on a seeded epsilon-free pair")
def test_compose_seeded(sub):
    compose = has(sub, "compose")
    rng = np.random.default_rng(0)
    labels = ["p", "q", "r"]

    def random_fst():
        arcs = [
            (0, 1, "p", "p", 0.2),
            (1, 2, "q", "q", 0.3),
            (2, 3, "r", "r", 0.1),
        ]
        for _ in range(4):
            src = int(rng.integers(0, 3))
            dst = int(rng.integers(src + 1, 4))
            label = labels[int(rng.integers(0, 3))]
            weight = float(rng.integers(1, 6)) / 10
            arcs.append((src, dst, label, label, weight))
        return _fst(arcs, {3: float(rng.integers(0, 4)) / 10})

    first, second = random_fst(), random_fst()
    want = _pair_paths(first, second)
    if not want:
        fail("the seeded pair produced no aligned path, so the check cannot tell compose from an empty graph")
    _same(compose(first, second), want, "seeded epsilon-free pair")


@check("shortest_path is the minimum sum of costs, not the maximum")
def test_shortest_path(sub):
    # shortest_path_add_probabilities keeps 0.9 here. The min of the sums is 0.4.
    shortest_path = has(sub, "shortest_path")
    graph = _fst(
        [
            (0, 1, "a", "a", 0.2),
            (1, 2, "b", "b", 0.2),
            (0, 2, "c", "c", 0.9),
        ],
        {2: 0.0},
    )
    cost, inputs, outputs = shortest_path(graph)
    if abs(float(cost) - 0.4) > 1e-6 or tuple(inputs) != ("a", "b") or tuple(outputs) != ("a", "b"):
        fail(
            "two-path graph: expected cost 0.4 on ('a', 'b') -> ('a', 'b'), "
            f"got cost {cost} on {tuple(inputs)} -> {tuple(outputs)}"
        )
    # Forgetting the final weight picks 'a' at 0.2. The min of the full sums is 'b' at 0.6.
    weighted = _fst([(0, 1, "a", "a", 0.2), (0, 2, "b", "b", 0.5)], {1: 1.0, 2: 0.1})
    cost, inputs, outputs = shortest_path(weighted)
    if abs(float(cost) - 0.6) > 1e-6 or tuple(outputs) != ("b",):
        fail(
            "final weights: expected cost 0.6 on output ('b',), "
            f"got cost {cost} on {tuple(outputs)}"
        )


@check("n_best returns the cheapest paths in order")
def test_n_best(sub):
    n_best = has(sub, "n_best")
    got = n_best(compose_ready(sub), 2)
    want_costs = [1.1, 1.4]
    if len(got) != 2:
        fail(f"n_best of the lexicon graph: expected 2 paths, got {len(got)}: {got}")
    for index, (path, want) in enumerate(zip(got, want_costs)):
        cost = float(path[0])
        if abs(cost - want) > 1e-6:
            fail(f"n_best path {index}: expected cost {want}, got {cost} from {got}")
    if tuple(got[0][2]) != ("a",) or tuple(got[1][2]) != ("cat",):
        fail(f"n_best outputs: expected ('a',) then ('cat',), got {tuple(got[0][2])} then {tuple(got[1][2])}")


def compose_ready(sub):
    """The lesson's lexicon composed with its bigram, through the reader's compose."""
    return has(sub, "compose")(_lexicon(), _grammar())


@check("total_weight is the log-semiring sum of every path")
def test_total_weight(sub):
    total_weight = has(sub, "total_weight")
    graph = compose_ready(sub)
    costs = [cost for cost, _, _ in _paths(graph)]
    want = float(-np.logaddexp.reduce(-np.array(costs, dtype=float)))
    got = float(total_weight(graph))
    if not np.isfinite(got) or abs(got - want) > 1e-6:
        fail(f"log-semiring total: expected {want:.6f}, got {got}")
    # A one-path graph is its own total. Summing the two arc weights as a
    # probability, 0.2 + 0.5 = 0.7, is not the log-semiring total 0.5.
    single = _fst([(0, 1, "a", "a", 0.2), (1, 2, "b", "b", 0.3)], {2: 0.0})
    got = float(total_weight(single))
    if abs(got - 0.5) > 1e-6:
        fail(f"one path of cost 0.5: expected total 0.5, got {got}")


def _translate(fst, symbols, limit=8):
    """Every (outputs, cost) the graph writes for exactly these input symbols.

    Cycles are fine because each step with a real input consumes a symbol.
    At most `limit` input-epsilon steps in a row are followed.
    """
    outgoing = _outgoing(fst)
    found = []

    def walk(state, index, cost, outputs, eps_run):
        if index == len(symbols) and state in fst["finals"]:
            found.append((outputs, round(cost + float(fst["finals"][state]), 6)))
        for dst, ilabel, olabel, weight in outgoing.get(state, []):
            nout = outputs if olabel == EPS else outputs + (olabel,)
            if ilabel == EPS and eps_run < limit:
                walk(dst, index, cost + weight, nout, eps_run + 1)
            elif index < len(symbols) and ilabel == symbols[index]:
                walk(dst, index + 1, cost + weight, nout, 0)

    walk(fst["start"], 0, 0.0, (), 0)
    return sorted(found)


def _expect(fst, symbols, want, label):
    got = _translate(fst, tuple(symbols))
    want = sorted((tuple(outputs), round(cost, 6)) for outputs, cost in want)
    if got != want:
        fail(f"{label}: reading {tuple(symbols)} expected {want}, got {got}")


@check("context_transducer writes the middle phone of each triphone, and only where the windows overlap")
def test_context_transducer(sub):
    # context_free_c accepts the two inconsistent strings below.
    context_transducer = has(sub, "context_transducer")
    c = context_transducer(["ah", "k", "t"])
    _expect(c, ["sil-k+t", "k-t+sil"], [(("k", "t"), 0.0)], "cat")
    _expect(c, ["sil-ah+sil"], [(("ah",), 0.0)], "one phone")
    _expect(c, ["sil-k+k", "k-k+t", "k-t+sil"], [(("k", "k", "t"), 0.0)], "a repeated phone")
    _expect(c, ["sil-k+t", "ah-t+sil"], [], "second window does not overlap the first")
    _expect(c, ["ah-k+t", "k-t+sil"], [], "no silence before the first phone")
    _expect(c, ["sil-k+t", "k-t+t"], [], "no silence after the last phone")
    _expect(c, ["sil-k+t"], [], "stops before the right context is silence")
    sp = context_transducer(["a", "b"], boundary="sp")
    _expect(sp, ["sp-a+b", "a-b+sp"], [(("a", "b"), 0.0)], "boundary sp")
    _expect(sp, ["sil-a+b", "a-b+sil"], [], "boundary sil is not sp")


@check("hmm_transducer reads senones as triphones, and hclg composes H, C, L, and G in that order")
def test_hclg(sub):
    # hclg_reversed composes G, L, C, and H, which leaves no path at all.
    hmm_transducer, context_transducer, hclg = (
        has(sub, "hmm_transducer"),
        has(sub, "context_transducer"),
        has(sub, "hclg"),
    )
    senones = {"sil-ah+sil": (0, 1), "sil-k+t": (2, 3), "k-t+sil": (4, 1)}
    h = hmm_transducer(senones)
    _expect(h, [0, 1], [(("sil-ah+sil",), 0.0)], "H on one triphone")
    _expect(h, [2, 3, 4, 1], [(("sil-k+t", "k-t+sil"), 0.0)], "H on two triphones")
    _expect(h, [4, 1, 0, 1], [(("k-t+sil", "sil-ah+sil"), 0.0)], "H on a shared senone")
    _expect(h, [2, 1], [], "senone 1 does not finish sil-k+t")
    _expect(h, [2], [], "a triphone needs all its senones")
    full = hclg(h, context_transducer(["ah", "k", "t"]), _lexicon(), _grammar())
    _expect(full, [2, 3, 4, 1], [(("cat",), 1.4)], "senones of cat")
    _expect(full, [0, 1], [(("a",), 1.1)], "senones of a")
    _expect(full, [2, 3, 0, 1], [], "triphone windows that do not overlap")


def _collapse():
    """The collapse function from the CTC chapter, found by its slug name."""
    import speechcheck

    runner = Path(speechcheck.__file__).resolve().parent
    roots = []
    if len(runner.parents) > 3:
        roots.append(runner.parents[3])
    env = os.environ.get("PYTHONPATH", "")
    roots.extend(Path(item) for item in env.split(os.pathsep) if item)
    for root in roots:
        matches = sorted(root.glob("content/exercises/*-ctc-many-paths-one-label/solution.py"))
        if len(matches) == 1:
            spec = importlib.util.spec_from_file_location("ctc_collapse_lesson", matches[0])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.collapse
    fail(
        "could not find the CTC chapter's collapse function under "
        "content/exercises/*-ctc-many-paths-one-label/solution.py"
    )


def _accepts(fst, symbols):
    outgoing = _outgoing(fst)
    seen = set()
    stack = [(fst["start"], 0)]
    while stack:
        state, index = stack.pop()
        key = (state, index)
        if key in seen:
            continue
        seen.add(key)
        if index == len(symbols) and state in fst["finals"]:
            return True
        if index == len(symbols):
            continue
        for dst, ilabel, _, _ in outgoing.get(state, []):
            if ilabel == symbols[index]:
                stack.append((dst, index + 1))
    return False


@check("ctc_topology accepts exactly the strings that collapse to the target")
def test_ctc_topology(sub):
    # ctc_topology_skip_repeat_blank accepts (1, 1) for target [1, 1].
    # collapse((1, 1)) is [1], so that string must be rejected.
    ctc_topology = has(sub, "ctc_topology")
    collapse = _collapse()
    cases = [
        ([1], [0, 1], 4),
        ([1, 1], [0, 1], 5),
        ([1, 2], [0, 1, 2], 4),
        ([1, 2, 2], [0, 1, 2], 4),
    ]
    for target, alphabet, longest in cases:
        topo = ctc_topology(target, blank=0)
        for length in range(longest + 1):
            for symbols in np.ndindex(*([len(alphabet)] * length)):
                string = tuple(alphabet[i] for i in symbols)
                got = _accepts(topo, string)
                collapsed = list(collapse(string, blank=0))
                want = collapsed == list(target)
                if got != want:
                    fail(
                        f"target {target}: string {string} collapses to {collapsed}, "
                        f"expected accepted={want}, got accepted={got}"
                    )
