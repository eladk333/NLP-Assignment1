import heapq
from collections import Counter, defaultdict

EOW = "§"
LITERAL_EOW = "[§]"


def count_words(filename):
    counts = Counter()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            counts.update(line.split())
    return counts


def merge_word(word, a, b, c):
    out = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            out.append(c)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def train_bpe(filename, N):
    word_counts = count_words(filename)

    # symbol 0 is the end of word marker, so a literal § in the text is a normal character
    chars = sorted({ch for w in word_counts for ch in w})
    vocab = [EOW] + [LITERAL_EOW if ch == EOW else ch for ch in chars]
    char_id = {ch: i + 1 for i, ch in enumerate(chars)}

    words = [[char_id[ch] for ch in w] + [0] for w in word_counts]
    freqs = list(word_counts.values())

    pair_counts = defaultdict(int)
    where = defaultdict(set)
    for i, w in enumerate(words):
        for pair in zip(w, w[1:]):
            pair_counts[pair] += freqs[i]
            where[pair].add(i)

    heap = [(-c, a, b) for (a, b), c in pair_counts.items()]
    heapq.heapify(heap)

    while len(vocab) < N and heap:
        neg, a, b = heapq.heappop(heap)
        if pair_counts.get((a, b), 0) != -neg:
            continue

        c = len(vocab)
        vocab.append(vocab[a] + vocab[b])
        changed = set()
        for i in where.pop((a, b)):
            old = words[i]
            new = merge_word(old, a, b, c)
            if len(new) == len(old):
                continue
            old_pairs = list(zip(old, old[1:]))
            new_pairs = list(zip(new, new[1:]))
            for pair in old_pairs:
                pair_counts[pair] -= freqs[i]
                changed.add(pair)
            for pair in new_pairs:
                pair_counts[pair] += freqs[i]
                changed.add(pair)
                where[pair].add(i)
            for pair in set(old_pairs) - set(new_pairs):
                if pair in where:
                    where[pair].discard(i)
            words[i] = new

        del pair_counts[(a, b)]
        changed.discard((a, b))
        for pair in changed:
            if pair_counts[pair] > 0:
                heapq.heappush(heap, (-pair_counts[pair], pair[0], pair[1]))
            else:
                del pair_counts[pair]
                where.pop(pair, None)

    return vocab


def tokenize(text, vocab):
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
            tokens.append(s[i:j])
            i = j
    return tokens


if __name__ == "__main__":
    for lang in ["english", "hebrew"]:
        vocab = train_bpe(f"data/{lang}.txt", 30000)
        with open(f"{lang}.bpe.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(vocab) + "\n")
