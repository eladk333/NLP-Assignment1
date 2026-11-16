"""
Byte-Pair Encoding (BPE) vocabulary training, following Sennrich et al. 2016
(https://arxiv.org/pdf/1508.07909) and the assignment's description.

The text is first split on whitespace into "classic tokens" (runs of
whitespace collapse to nothing, as in str.split()). Each classic token is
represented as a word made of its characters plus a trailing end-of-word
symbol EOW ("§"). Since EOW is always the last symbol of a word, it can
only ever be merged with the symbol before it, never with a symbol after it
- which is exactly the constraint the assignment asks for ("don't allow
merging a whitespace with items after it, only with items before it").

Training works on word-TYPES weighted by their frequency (not on every
occurrence in the text), and keeps an index from each adjacent pair to the
set of word-types containing it, so that merging the current best pair only
touches the (typically small) set of word-types affected by it, instead of
rescanning the whole corpus on every merge.

Usage: python bpe.py <filename> <N> [output_file]
"""
import heapq
import sys
import time
from collections import Counter, defaultdict

EOW = "§"  # "end of word" marker, standing in for whitespace


def _word_counts(filename):
    """Count classic-token (whitespace-split) frequencies in the file."""
    counts = Counter()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            counts.update(line.split())
    return counts


def _init_words(word_counts):
    """Turn {word: freq} into {symbols-tuple: freq}, and the initial vocab."""
    vocab = set()
    words = {}
    for word, freq in word_counts.items():
        symbols = tuple(word) + (EOW,)
        vocab.update(symbols)
        words[symbols] = words.get(symbols, 0) + freq
    return vocab, words


def _count_pairs(words):
    """Initial adjacent-pair counts, and which word-types contain each pair."""
    pair_freq = Counter()
    pair_to_words = defaultdict(set)
    for word, freq in words.items():
        for pair in zip(word, word[1:]):
            pair_freq[pair] += freq
            pair_to_words[pair].add(word)
    return pair_freq, pair_to_words


def _apply_merge(word, a, b, merged):
    """Return `word` with every non-overlapping (a, b) -> merged applied."""
    new_word = []
    i = 0
    n = len(word)
    while i < n:
        if i + 1 < n and word[i] == a and word[i + 1] == b:
            new_word.append(merged)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def train_bpe(filename, N):
    """Train a BPE vocabulary of size N on the text in `filename`.

    Returns a list of N strings: the initial per-character vocabulary plus
    the merges learned, in the order they were created.
    """
    word_counts = _word_counts(filename)
    vocab, words = _init_words(word_counts)
    vocab_list = sorted(vocab)

    if len(vocab_list) >= N:
        return vocab_list[:N]

    pair_freq, pair_to_words = _count_pairs(words)

    # Max-heap (via negated counts) of candidate pairs, with lazy deletion:
    # entries may be stale, we check them against pair_freq when popped.
    heap = [(-freq, pair) for pair, freq in pair_freq.items()]
    heapq.heapify(heap)

    while len(vocab_list) < N:
        # Find the current best valid pair.
        pair = None
        while heap:
            neg_freq, candidate = heapq.heappop(heap)
            if pair_freq.get(candidate) == -neg_freq and -neg_freq > 0:
                pair = candidate
                break
        if pair is None:
            break  # no more pairs to merge

        a, b = pair
        merged = a + b
        vocab_list.append(merged)

        affected = pair_to_words.pop(pair, ())
        for word in list(affected):
            freq = words.pop(word, None)
            if freq is None:
                continue

            # Remove this word's contribution to all its pairs.
            for p in zip(word, word[1:]):
                pair_freq[p] -= freq
                if pair_freq[p] <= 0:
                    del pair_freq[p]
                pair_to_words[p].discard(word)

            new_word = _apply_merge(word, a, b, merged)
            words[new_word] = words.get(new_word, 0) + freq

            # Add the new word's contribution to all its pairs.
            for p in zip(new_word, new_word[1:]):
                pair_freq[p] += freq
                pair_to_words[p].add(new_word)
                heapq.heappush(heap, (-pair_freq[p], p))

    return vocab_list


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # so Hebrew prints on Windows
    path = sys.argv[1]
    n = int(sys.argv[2])
    out_path = sys.argv[3] if len(sys.argv) > 3 else None

    t0 = time.time()
    vocab = train_bpe(path, n)
    print(f"Trained {len(vocab)} vocab items in {time.time() - t0:.1f}s", flush=True)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for v in vocab:
                f.write(v + "\n")
    else:
        for v in vocab[:50]:
            print(repr(v))
