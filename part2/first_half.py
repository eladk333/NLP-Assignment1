"""Question 4: compare the types in the first half of a file with the whole file.

Uses the full counts file written by count_tokens.py, so only the first half
of the text needs to be read: second-half counts = full counts - first-half counts.

Usage:
    python first_half.py data/english.txt english.counts.tsv
"""
import argparse
import sys
from collections import Counter

from count_tokens import open_text, tokenize


def read_counts(filename):
    counts = Counter()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            token, c = line.rstrip("\n").split("\t")
            counts[token] = int(c)
    return counts


def count_first_tokens(filename, n):
    counts = Counter()
    seen = 0
    with open_text(filename) as f:
        for line in f:
            for token in tokenize(line):
                if seen == n:
                    return counts
                counts[token] += 1
                seen += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("filename", help="text file (.txt or .txt.gz)")
    parser.add_argument("counts", help="full counts TSV from count_tokens.py")
    parser.add_argument("-k", "--examples", type=int, default=10, help="examples to show per threshold")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252

    full = read_counts(args.counts)
    n_tokens = sum(full.values())
    first = count_first_tokens(args.filename, n_tokens // 2)

    print(f"tokens: {n_tokens}, first half: {n_tokens // 2}")
    print(f"types in full file:  {len(full)}")
    print(f"types in first half: {len(first)}")
    print(f"types added in second half: {len(full) - len(first)}")

    # types that never appear in the first half; all their occurrences are in the second half
    new_types = Counter({t: c for t, c in full.items() if t not in first})
    for threshold in (2, 5, 10):
        above = [(t, c) for t, c in new_types.most_common() if c > threshold]
        print(f"\nnew types appearing more than {threshold} times in second half: {len(above)}")
        for t, c in above[:args.examples]:
            print(f"  {t}\t{c}")
        if len(above) > args.examples:
            # also show some examples near the threshold, which are more typical
            print("  ... near the threshold:")
            for t, c in above[-args.examples:]:
                print(f"  {t}\t{c}")


if __name__ == "__main__":
    main()
