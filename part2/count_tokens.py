"""Count tokens and token-types in a text file.

A token is a whitespace-separated sequence with leading/trailing punctuation
stripped, so word-internal punctuation (e.g. "ice-cream") is kept.

Usage:
    python count_tokens.py data/english.txt
    python count_tokens.py data/hebrew.txt -o hebrew.counts.tsv
"""
import argparse
import gzip
import sys
import unicodedata
from collections import Counter


def is_punct(ch):
    # Unicode categories P* (punctuation); symbols such as $, +, | are kept
    return unicodedata.category(ch)[0] == "P"


def strip_punct(word):
    start, end = 0, len(word)
    while start < end and is_punct(word[start]):
        start += 1
    while end > start and is_punct(word[end - 1]):
        end -= 1
    return word[start:end]


def tokenize(line):
    for word in line.split():
        token = strip_punct(word)
        if token:
            yield token


def open_text(filename):
    if filename.endswith(".gz"):
        return gzip.open(filename, "rt", encoding="utf-8")
    return open(filename, encoding="utf-8")


def count_tokens(filename):
    counts = Counter()
    with open_text(filename) as f:
        for line in f:
            counts.update(tokenize(line))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("filename", help="text file (.txt or .txt.gz)")
    parser.add_argument("-o", "--output", help="write type counts as TSV, most frequent first")
    args = parser.parse_args()

    counts = count_tokens(args.filename)
    n_tokens = sum(counts.values())
    n_types = len(counts)
    print(f"tokens: {n_tokens}")
    print(f"types:  {n_types}")
    print(f"tokens/types: {n_tokens / n_types:.2f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as out:
            for token, c in counts.most_common():
                out.write(f"{token}\t{c}\n")
        print(f"wrote counts to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
