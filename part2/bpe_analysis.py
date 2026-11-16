"""Inspect the trained BPE vocabularies for the bpe.pdf questions.

Q1 counts the vocabulary items that are complete tokens in the sense of part 1:
an item `w§` (a full word followed by the end-of-word marker) where `w`, with
leading/trailing punctuation stripped, is a token-type of the corpus.

Usage:
    python bpe_analysis.py english
    python bpe_analysis.py hebrew --split word1 word2 ...
"""
import argparse
from collections import defaultdict

from bpe import EOW, LITERAL_EOW, tokenize
from count_tokens import strip_punct


def load_vocab(lang):
    with open(f"{lang}.bpe.txt", encoding="utf-8") as f:
        return f.read().split("\n")[:-1]


def load_types(lang):
    types = {}
    with open(f"{lang}.counts.tsv", encoding="utf-8") as f:
        for line in f:
            token, count = line.rstrip("\n").split("\t")
            types[token] = int(count)
    return types


def complete_tokens(vocab, types):
    """Map each complete token to the vocabulary items that spell it (e.g. the§, the,§)."""
    found = defaultdict(list)
    for item in vocab:
        if item.endswith(EOW) and item != EOW:
            token = strip_punct(item[:-1].replace(LITERAL_EOW, "§"))
            if token in types:
                found[token].append(item)
    return found


def standalone_share(found, types):
    """For each token, the fraction of its word-final occurrences that are the whole word.

    "ing" occurs once as a standalone token, but millions of times as the end of
    another word, so `ing§` is really a suffix; "the" is almost always itself.
    """
    max_len = max(map(len, found))
    ending = defaultdict(int)
    for t, c in types.items():
        for k in range(1, min(len(t), max_len) + 1):
            if t[-k:] in found:
                ending[t[-k:]] += c
    return {t: types[t] / ending[t] for t in found}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lang", choices=["english", "hebrew"])
    parser.add_argument("--split", nargs="+", help="show how these words are tokenized")
    args = parser.parse_args()

    vocab = load_vocab(args.lang)
    if args.split:
        for word in args.split:
            print(word, "->", " ".join(tokenize(word, vocab)))
        return

    types = load_types(args.lang)
    found = complete_tokens(vocab, types)
    bare = sum(1 for t, items in found.items() if t + EOW in items)
    print(f"vocabulary items: {len(vocab)}")
    print(f"items ending with {EOW}: {sum(1 for v in vocab if v.endswith(EOW)) - 1}")
    print(f"items that are a complete token (punctuation allowed): {sum(map(len, found.values()))}")
    print(f"  distinct complete tokens: {len(found)}")
    print(f"  complete tokens without punctuation (w{EOW}): {bare}")
    multi = sorted(found.items(), key=lambda kv: -len(kv[1]))[:8]
    print("  tokens with several spellings:", "; ".join(" ".join(items) for _, items in multi))
    share = standalone_share(found, types)
    for threshold in (0.1, 0.5, 0.9):
        words = [t for t in found if share[t] >= threshold]
        print(f"  standalone share >= {threshold}: {len(words)} tokens, "
              f"{sum(len(found[t]) for t in words)} items")
    border = sorted((t for t in found if 0.3 <= share[t] <= 0.7), key=types.get, reverse=True)
    print("  borderline (share 0.3-0.7):", ", ".join(f"{t}({share[t]:.2f})" for t in border[:15]))
    suffixes = sorted((t for t in found if share[t] < 0.5), key=lambda t: share[t])
    print("  rejected as suffixes:", ", ".join(f"{t}({share[t]:.4f})" for t in suffixes[:15]))


if __name__ == "__main__":
    main()
