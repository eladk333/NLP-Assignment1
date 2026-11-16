"""N-gram statistics: top-k 5-grams and 10-grams, and the longest n-gram that appears more than once.

Storing every n-gram as a tuple of strings does not fit in memory for ~85M tokens,
so tokens are mapped to integer ids and each n-gram is represented by a 64-bit
polynomial hash computed with numpy. Every reported n-gram is verified against the
actual token ids, so a hash collision cannot produce a wrong answer silently.

By default the file is treated as one continuous token stream, so n-grams may cross
line breaks. With --within-lines, n-grams never cross a line break: each break gets a
hash unique to its position, so any n-gram containing it occurs only once.

Ties are broken alphabetically, so the output is deterministic.

Usage:
    python ngrams.py data/english.txt
    python ngrams.py data/hebrew.txt --within-lines -o hebrew.longest.txt
"""
import argparse
import os
import sys
from array import array

import numpy as np

from count_tokens import open_text, tokenize

BASE = 0x100000001B3  # odd multiplier for the polynomial hash (mod 2**64)
LINE_BREAK = -1  # id marking the end of each line in the cached stream
MAX_TIE_SORT = 1000  # largest number of tied n-grams to order alphabetically


def load_ids(filename, cache_dir="cache"):
    """Return (ids, vocab): the token stream as int32 ids with LINE_BREAK after each line, and id -> token."""
    stem = os.path.join(cache_dir, os.path.basename(filename).split(".")[0])
    if os.path.exists(stem + ".ids_lines.npy"):
        with open(stem + ".vocab.txt", encoding="utf-8") as f:
            vocab = f.read().split("\n")
        return np.load(stem + ".ids_lines.npy"), vocab

    index = {}
    ids = array("i")
    with open_text(filename) as f:
        for line in f:
            for token in tokenize(line):
                i = index.get(token)
                if i is None:
                    i = index[token] = len(index)
                ids.append(i)
            ids.append(LINE_BREAK)
    ids = np.frombuffer(ids, dtype=np.int32)
    vocab = list(index)  # dicts keep insertion order, so position == id

    os.makedirs(cache_dir, exist_ok=True)
    np.save(stem + ".ids_lines.npy", ids)
    with open(stem + ".vocab.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))  # tokens never contain whitespace
    return ids, vocab


def token_hashes(ids):
    """Scramble ids into well-spread 64-bit values (splitmix64 finalizer)."""
    x = (ids.astype(np.uint64) + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def ngram_hashes(h1, n):
    """Hash of every n-gram: result[i] covers tokens i..i+n-1 (length len(h1)-n+1).

    Built by binary decomposition of n: blocks of length 1, 2, 4, ... are combined
    with hash(A + B) = hash(A) * BASE**len(B) + hash(B), so it takes O(log n) passes.
    """
    total = len(h1)
    result, rlen = None, 0
    block, blen = h1, 1
    m = n
    while True:
        if m & 1:
            if result is None:
                result, rlen = block.copy(), blen
            else:
                size = total - (rlen + blen) + 1
                result = result[:size] * np.uint64(pow(BASE, blen, 2**64)) + block[rlen:rlen + size]
                rlen += blen
        m >>= 1
        if not m:
            return result
        block = block[:-blen] * np.uint64(pow(BASE, blen, 2**64)) + block[blen:]
        blen *= 2


def repeated_counts(hashes):
    """Sort hashes in place; return (values, counts) of hashes that occur more than once."""
    hashes.sort()
    starts = np.flatnonzero(np.concatenate(([True], hashes[1:] != hashes[:-1])))
    counts = np.diff(np.append(starts, len(hashes)))
    keep = counts > 1
    return hashes[starts[keep]], counts[keep]


def verified(ids, n, positions):
    """Check that every occurrence of an n-gram hash is the same tokens, with no line break inside."""
    windows = ids[positions[:, None] + np.arange(n)]
    if not (windows == windows[0]).all() or (windows[0] == LINE_BREAK).any():
        raise RuntimeError(f"hash collision for a {n}-gram")
    return positions


def occurrences(ids, h1, n, values):
    """Verified positions of each n-gram hash in values: {value: positions}."""
    hashes = ngram_hashes(h1, n)
    positions = np.flatnonzero(np.isin(hashes, values))
    found = hashes[positions]
    return {v: verified(ids, n, positions[found == v]) for v in values}


def top_ngrams(ids, h1, n, k, text):
    """Return ([(text, count)] for the k most frequent n-grams, size of the tie at place k).

    Equal counts are ordered alphabetically when there are at most MAX_TIE_SORT candidates.
    """
    values, counts = repeated_counts(ngram_hashes(h1, n))
    if not len(counts):
        return [], 0
    order = np.argsort(-counts, kind="stable")
    kth = counts[order[min(k, len(order)) - 1]]
    candidates = order[:min(int((counts >= kth).sum()), k + MAX_TIE_SORT)]
    positions = occurrences(ids, h1, n, values[candidates])
    rows = []
    for j in candidates:
        p = positions[values[j]]
        assert len(p) == counts[j]
        rows.append((text(ids[p[0]:p[0] + n]), int(counts[j])))
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows[:k], int((counts == kth).sum())


def has_repeat(h1, n):
    hashes = ngram_hashes(h1, n)
    hashes.sort()
    return bool((hashes[1:] == hashes[:-1]).any())


def longest_repeated(ids, h1):
    """Largest n such that some n-gram appears more than once (exponential + binary search).

    If an n-gram repeats, so does each of its (n-1)-grams, so the property is monotone in n.
    """
    lo, hi = 1, 2  # invariant: has_repeat(lo) and not has_repeat(hi)
    while has_repeat(h1, hi):
        lo, hi = hi, min(hi * 2, len(ids))
        print(f"  repeat exists at n={lo}", file=sys.stderr)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if has_repeat(h1, mid):
            lo = mid
        else:
            hi = mid
        print(f"  searching n in [{lo}, {hi})", file=sys.stderr)

    values, counts = repeated_counts(ngram_hashes(h1, lo))
    positions = occurrences(ids, h1, lo, values)
    return lo, [(positions[v], int(c)) for v, c in zip(values, counts)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("filename", help="text file (.txt or .txt.gz)")
    parser.add_argument("-k", type=int, default=10, help="how many top n-grams to show")
    parser.add_argument("-o", "--output", help="write the full longest repeated n-gram(s) here")
    parser.add_argument("--within-lines", action="store_true", help="do not let n-grams cross line breaks")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252

    ids, vocab = load_ids(args.filename)
    breaks = np.flatnonzero(ids == LINE_BREAK)
    if args.within_lines:
        h1 = token_hashes(ids)
        # ids above the vocabulary range, so each break hashes like a token seen only once
        h1[breaks] = token_hashes(breaks.astype(np.int64) + 2**33)
    else:
        ids = np.delete(ids, breaks)
        h1 = token_hashes(ids)
    print(f"tokens: {len(ids) - (len(breaks) if args.within_lines else 0)}")
    print("n-grams " + ("stay within a line" if args.within_lines else "may cross line breaks"))

    def text(token_ids):
        return " ".join(vocab[i] for i in token_ids)

    for n in (5, 10):
        top, tied = top_ngrams(ids, h1, n, args.k, text)
        print(f"\ntop {args.k} {n}-grams:")
        for rank, (words, count) in enumerate(top, 1):
            print(f"{rank:3}\t{count}\t{words}")
        if tied > 1:
            print(f"  ({tied} {n}-grams share the count at place {args.k})")

    n, found = longest_repeated(ids, h1)
    print(f"\nlongest n-gram with frequency > 1: n={n}, {len(found)} such n-gram(s)")
    out = open(args.output, "w", encoding="utf-8") if args.output else None
    for positions, count in found:
        words = text(ids[positions[0]:positions[0] + n])
        preview = words if n <= 60 else " ".join(words.split()[:60]) + " ..."
        print(f"  frequency {count}, at token positions {positions.tolist()}:\n  {preview}")
        if out:
            out.write(f"n={n} frequency={count} positions={positions.tolist()}\n{words}\n\n")
    if out:
        out.close()
        print(f"wrote full text to {args.output}")


if __name__ == "__main__":
    main()
