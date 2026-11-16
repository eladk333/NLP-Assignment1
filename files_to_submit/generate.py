from collections import defaultdict
import random

class PCFG(object):
    def __init__(self):
        self._rules = defaultdict(list)
        self._sums = defaultdict(float)

    def add_rule(self, lhs, rhs, weight):
        assert(isinstance(lhs, str))
        assert(isinstance(rhs, list))
        self._rules[lhs].append((rhs, weight))
        self._sums[lhs] += weight

    @classmethod
    def from_file(cls, filename):
        grammar = PCFG()
        with open(filename) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if not line: continue
                w,l,r = line.split(None, 2)
                r = r.split()
                w = float(w)
                grammar.add_rule(l,r,w)
        return grammar

    def is_terminal(self, symbol): return symbol not in self._rules

    def gen(self, symbol, tree=False):
        """
        Expands symbol into a random string of terminals. With tree=True,
        every nonterminal's expansion is wrapped as "(SYMBOL children...)",
        giving the bracketed derivation tree instead of the plain sentence.
        """
        if self.is_terminal(symbol): return symbol
        else:
            expansion = self.random_expansion(symbol)
            children = " ".join(self.gen(s, tree) for s in expansion)
            return "(%s %s)" % (symbol, children) if tree else children

    def random_sent(self, tree=False):
        return self.gen("ROOT", tree)

    def random_expansion(self, symbol):
        """
        Generates a random RHS for symbol, in proportion to the weights.
        """
        p = random.random() * self._sums[symbol]
        for r,w in self._rules[symbol]:
            p = p - w
            if p < 0: return r
        return r


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Generate random sentences from a PCFG.")
    parser.add_argument("grammar", help="path to the grammar file")
    parser.add_argument("-n", type=int, default=1, help="number of sentences to generate (default: 1)")
    parser.add_argument("-t", action="store_true", help="print the derivation tree of each sentence")
    args = parser.parse_args()

    pcfg = PCFG.from_file(args.grammar)
    for _ in range(args.n):
        print(pcfg.random_sent(tree=args.t))