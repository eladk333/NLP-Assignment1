"""Byte-pair encoding (BPE): vocabulary training and greedy tokenization.

The text is split on whitespace into "classic" words, and every word gets an
end-of-word marker. The marker is written as `§` in the output vocabulary, so it
only ever appears standalone or at the end of an item, and merges never cross
word boundaries.

Training runs on word-type counts instead of the raw text: each pair is counted
once per type, weighted by the type's frequency. Pair counts are kept
incrementally (only the words containing the merged pair are revisited), and the
most frequent pair is found with a max-heap with lazy deletion.

`§` does appear in the data (e.g. "18 U.S.C. § 922"). Internally the end-of-word
marker is a separate symbol, so a literal `§` is just another character; in the
output it is written as `[§]`, which keeps every vocabulary string unambiguous.

Usage:
    python bpe.py data/english.txt 30000 -o english.bpe.txt
    python bpe.py data/hebrew.txt.gz 30000 -o hebrew.bpe.txt
"""
import argparse
import gzip
import heapq
import sys
import time
from collections import Counter, defaultdict

EOW = "§"
LITERAL_EOW = "[§]"


def open_text(filename):
    if filename.endswith(".gz"):
        return gzip.open(filename, "rt", encoding="utf-8")
    return open(filename, encoding="utf-8")


def count_words(filename):
    """Count whitespace-separated words (punctuation is kept, unlike part 1)."""
    counts = Counter()
    with open_text(filename) as f:
        for line in f:
            counts.update(line.split())
    return counts


def merge_word(word, a, b, c):
    """Replace every non-overlapping (a, b) in `word`, left to right, with c."""
    out = []
    i, n = 0, len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(c)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def train_bpe_from_counts(word_counts, N, verbose=False):
    """Learn a BPE vocabulary of N strings from a {word: frequency} mapping."""
    # Symbols are ints: 0 is the end-of-word marker, then one id per character.
    chars = sorted({ch for w in word_counts for ch in w})
    vocab = [EOW] + [LITERAL_EOW if ch == EOW else ch for ch in chars]
    char_id = {ch: i + 1 for i, ch in enumerate(chars)}
    if N < len(vocab):
        raise ValueError(f"N={N} is smaller than the {len(vocab)} initial symbols")

    words = [[char_id[ch] for ch in w] + [0] for w in word_counts]
    freqs = list(word_counts.values())

    pair_counts = defaultdict(int)
    where = defaultdict(set)  # pair -> indices of the words that contain it
    for i, (w, f) in enumerate(zip(words, freqs)):
        for pair in zip(w, w[1:]):
            pair_counts[pair] += f
            where[pair].add(i)

    # Max-heap via negated counts; ties are broken by the smaller symbol ids.
    heap = [(-c, a, b) for (a, b), c in pair_counts.items()]
    heapq.heapify(heap)

    start = time.time()
    while len(vocab) < N and heap:
        neg, a, b = heapq.heappop(heap)
        if pair_counts.get((a, b), 0) != -neg:
            continue  # stale entry: the count changed after it was pushed

        c = len(vocab)
        vocab.append(vocab[a] + vocab[b])
        changed = set()
        for i in where.pop((a, b)):
            old = words[i]
            new = merge_word(old, a, b, c)
            if len(new) == len(old):
                continue
            f = freqs[i]
            old_pairs = list(zip(old, old[1:]))
            new_pairs = list(zip(new, new[1:]))
            for pair in old_pairs:
                pair_counts[pair] -= f
                changed.add(pair)
            for pair in new_pairs:
                pair_counts[pair] += f
                changed.add(pair)
                where[pair].add(i)
            for pair in set(old_pairs) - set(new_pairs):
                if pair in where:
                    where[pair].discard(i)
            words[i] = new

        del pair_counts[(a, b)]
        changed.discard((a, b))
        for pair in changed:
            count = pair_counts[pair]
            if count > 0:
                heapq.heappush(heap, (-count, pair[0], pair[1]))
            else:
                del pair_counts[pair]
                where.pop(pair, None)

        if verbose and len(vocab) % 1000 == 0:
            print(f"  {len(vocab)} items, last merge {vocab[a]!r} + {vocab[b]!r} "
                  f"({-neg} times), {time.time() - start:.0f}s", file=sys.stderr)

    if len(vocab) < N:
        print(f"warning: no pairs left to merge, vocabulary has only {len(vocab)} items",
              file=sys.stderr)
    return vocab


def train_bpe(filename, N):
    """Train BPE on a UTF-8 text file (.txt or .txt.gz) and return N vocabulary strings."""
    return train_bpe_from_counts(count_words(filename), N)


def tokenize(text, vocab):
    """Split text into vocabulary items, greedily taking the longest known prefix."""
    items = set(vocab)
    max_len = max(len(v) for v in vocab)
    tokens = []
    for word in text.split():
        s = word.replace(EOW, LITERAL_EOW) + EOW
        i = 0
        while i < len(s):
            for j in range(min(len(s), i + max_len), i, -1):
                if s[i:j] in items:
                    break
            # Characters missing from the vocabulary become single-char tokens.
            tokens.append(s[i:j])
            i = j
    return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("filename", help="text file (.txt or .txt.gz)")
    parser.add_argument("N", type=int, help="target vocabulary size")
    parser.add_argument("-o", "--output", help="write the vocabulary, one item per line")
    args = parser.parse_args()

    start = time.time()
    counts = count_words(args.filename)
    print(f"{sum(counts.values())} words, {len(counts)} word types "
          f"({time.time() - start:.0f}s)", file=sys.stderr)
    vocab = train_bpe_from_counts(counts, args.N, verbose=True)
    print(f"trained {len(vocab)} items in {time.time() - start:.0f}s", file=sys.stderr)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as out:
            out.write("\n".join(vocab) + "\n")  # items never contain whitespace
    else:
        print("\n".join(vocab))


if __name__ == "__main__":
    main()
