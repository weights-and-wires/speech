"""Numbers for the figure and worked examples in 077-pronunciation-assessment."""

import numpy as np

PHONES = ["k", "ae", "t", "s"]
POST = np.array([
    [0.80, 0.10, 0.05, 0.05],
    [0.70, 0.20, 0.05, 0.05],
    [0.10, 0.80, 0.05, 0.05],
    [0.10, 0.70, 0.10, 0.10],
    [0.05, 0.90, 0.025, 0.025],
    [0.05, 0.05, 0.20, 0.70],
    [0.10, 0.10, 0.30, 0.50],
])
SEGMENTS = [(0, 0, 2), (1, 2, 5), (2, 5, 7)]  # phone, start, end (exclusive)

print("row sums", POST.sum(axis=1))
for p, a, b in SEGMENTS:
    lp = np.log(POST[a:b, p])
    mx = np.log(POST[a:b].max(axis=1))
    print(PHONES[p], "frames", a, b,
          "lpp %.4f" % lp.mean(), "gop %.4f" % (lp - mx).mean(),
          "sum-gop %.4f" % (lp - mx).sum(),
          "post-ratio %.4f" % (POST[a:b, p] / POST[a:b].max(axis=1)).mean())

# the same t, stretched to 6 frames at the same posteriors
lp = np.log(np.tile([0.2, 0.3], 3)) - np.log(np.tile([0.7, 0.5], 3))
print("stretched t: mean %.4f sum %.4f" % (lp.mean(), lp.sum()))

# detection example
gop = np.array([-0.1, -0.4, -2.0, -0.3, -0.8, -1.5, -0.2, -2.5])
gold = np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)
for thr in (-1.0, -0.35):
    pred = gop < thr
    tp = (pred & gold).sum(); fp = (pred & ~gold).sum(); fn = (~pred & gold).sum()
    p = tp / (tp + fp); r = tp / (tp + fn)
    print("thr", thr, "TP FP FN", tp, fp, fn, "P %.4f R %.4f F1 %.4f" % (p, r, 2 * p * r / (p + r)))

# correlation example
x = np.array([-1.2, -0.9, -0.6, -0.5, -0.1])
h = np.array([3, 5, 5, 8, 9.0])
print("pearson %.4f" % np.corrcoef(x, h)[0, 1])
ranks_h = np.array([1, 2.5, 2.5, 4, 5])
print("spearman avg ranks %.4f" % np.corrcoef(np.arange(1, 6), ranks_h)[0, 1])
print("spearman ordinal ranks %.4f" % np.corrcoef(np.arange(1, 6), np.arange(1, 6))[0, 1])

# confusion example
can = np.array([2, 2, 0, 1, 2, 3])
rec = np.array([3, 2, 0, 1, 3, 3])
C = np.zeros((4, 4), int)
np.add.at(C, (can, rec), 1)
print(C)
