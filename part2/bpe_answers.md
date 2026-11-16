# Subword Tokenization with BPE

## Setup

- `bpe.py` implements `train_bpe(filename, N)`. It counts whitespace-separated word types once, then counts pairs per type weighted by the type's frequency. Pair counts are updated incrementally after every merge, and a max-heap finds the most frequent pair.
- Training was run with N = 30,000 on both files. English: 85.5M words, 1.76M word types, 5.5 minutes. Hebrew: 64.0M words, 2.95M word types, 7 minutes.
- Words are split on whitespace only, so punctuation stays inside the word (`apples,` → `a p p l e s , §`), as the README describes.
- **Handling `§` in the text.** The `§` character does appear in the data (e.g. "18 U.S.C. § 922": 299 times in English, 102 in Hebrew). Internally, the end-of-word marker is a separate symbol, so a literal `§` is treated as an ordinary character. In the output, the marker is written as `§` and a literal `§` is written as `[§]`, so no two vocabulary strings are the same.
- **The initial vocabulary is large.** The files contain 4,978 (English) and 5,506 (Hebrew) distinct characters, mostly rare characters from other scripts (CJK, Arabic, symbols). These all enter the vocabulary, so only about 25,000 of the 30,000 slots are learned merges.

## 1. How many complete tokens are in the vocabulary?

A vocabulary item is counted as a complete token if it has the form `w§` and `w` is a token-type of the corpus by the part-1 definition. Punctuation is stripped before the check, so `the,§` also counts as the token `the`.

| | English | Hebrew |
|---|---|---|
| Items of the exact form `w§` (no punctuation) | **11,211** | **14,439** |
| Items that spell a complete token, including punctuation variants | 14,899 | 17,464 |
| Distinct tokens covered by those items | 11,263 | 14,471 |
| Lower bound: the word appears on its own in at least 10% of the times it ends a word | 10,160 | 12,012 |

Notes:

- **The raw count is a little too high.** Some items are really suffixes that happen to appear alone a few times in the corpus. Examples: `ould§`, `ning§`, `ity§`, `ers§`, `ות§`, `עים§` (each appears 1 to a few dozen times standalone, but millions of times at the end of other words). The last row filters these out.
- **The filter is too strict for Hebrew.** Because prefixes attach to words, a real word like `עולם` mostly appears inside `העולם` or `בעולם`, so its standalone share is only 0.03–0.1. That is why the filtered Hebrew number is only a lower bound.
- **Hebrew has more complete tokens than English.** A Hebrew word with its prefixes is one token (`בשנת`, `לאחר`, `והוא`), so the frequent forms include many prefix+word combinations. Hebrew also has many more word types overall, so more distinct frequent forms exist.
- **Punctuation variants take up thousands of slots.** `ing§ ing,§ ing.§ ing"§ ing)§ ...` all appear: 3,688 extra items in English and 3,025 in Hebrew. A tokenizer that splits punctuation off first would free these slots for other items.

## 2. Subword items that make sense linguistically

**English.** Many of the earliest merges are real morphemes:

- Inflectional suffixes: `ed§` (merge #13), `ing§` (#30), `ly§` (#73), `s§`, `es§`, `ers§`, `ies§`.
- Derivational suffixes: `tion§` (#139), `ation§`, `ity§`, `ment§`, `ous§`, `able§`, `ist§`, `ful§`, `less§`, `ness§`, `ism§`, `ization§`.
- Prefixes: `re` (#20), `un` (#77), `pro`, `pre`, `dis`, `over`, `inter`, `anti`.

These make sense because the same affix attaches to thousands of different stems. Each stem may be rare, but the affix is very frequent, which is exactly what BPE rewards. The resulting splits often match the actual morphology:

- `un believ able§`
- `un even§`
- `re organized§`
- `dis like§`
- `anti dis establish ment ari anism§`
- `kind ness§`
- `king ly§`, `man ly§`

**Hebrew.** The very first merges are suffixes and prefixes:

- Suffixes: `ים§` (masculine plural, merge #8), `ות§` (feminine plural, #11), `ית§`, `יות§`, `ויות§`, `יים§`.
- Possessive suffixes: `ותו§`, `ותיה§`.
- Verb person suffixes: `תי§` (I, past tense), `נו§` (we).
- Prefix combinations: `ול` (#22), `וב` (#30) (ו "and" + ל/ב), `המ` (#15), `הת` (#44), `הש` (#58), `כש` ("when"), `לה`, `וה`.

Hebrew morphology is concatenative around the stem: prefixes ו/ה/ב/ל/מ/ש/כ and suffixes for number, gender and person. So these units recur across many different roots. Good splits:

- `כש הלכ תי§` = when + walk(ed) + I
- `שבת ות§` = Sabbath + plural
- `עלי ות§`, `זווי ות§` = stem + plural
- `נורבגי ת§`, `ביזנטי ת§` = adjective + feminine ending

## 3. Subword items that do not make sense linguistically

- **Spelling units rather than morphemes.** `ould§` (could/would/should), `igh` (high/night/light), `oug`, `ough§`, `clud` (include/exclude/conclude), `duc`, `bec`, `fir`. These are frequent letter sequences, so BPE merges them, but they carry no meaning of their own. `clud` is a borderline case: it is a Latin root, but not an English morpheme.
- **Truncated names and words.** `Scharnhor` (Scharnhorst), `Macbe` (Macbeth), `mariju` (marijuana), `Guatemal`, `catastro`, `Encyclop`, `Televisi`, `abdom`. In Hebrew: `ישרא` (merge #279), `הארצישרא`, `לייפצי` (Leipzig), `אינדיבידו`.
  - BPE builds long items left to right, and every intermediate step stays in the vocabulary.
  - These appear among the last merges (#24,000–#25,000), with frequencies around 600–700. The merge that would complete the word, e.g. `Scharnhor + st§`, did not happen before the vocabulary reached 30,000.
  - The leftover fragment is useful almost only inside that one word.
  - `ישרא` is an early example: the full word ישראל is very frequent, but it also appears inside many forms (ישראלי, ישראלית, הישראלי...), so the shared prefix was merged first.
- **Prefixes glued onto partial stems.** In Hebrew, merges cross the morpheme boundary, so items like `והבי`, `הבי`, `העי`, `בעי` combine a prefix with part of a stem.
- **Punctuation attached to suffixes.** Items like `ing,§`, `ed.§`, `,"§`, `."§` and `ם",§`. Here punctuation is merged with suffixes, which is pure spelling noise.

## 4. Items that make sense in some words and not in others

Yes. Often the same string is a morpheme in one word and a coincidental letter sequence in another. The greedy longest-prefix tokenizer cannot tell the two cases apart. It also sometimes picks a boundary that no merge sequence intended.

English:

| item | makes sense | does not make sense |
|---|---|---|
| `ness§` | `kind ness§`, `red ness§` | `har ness§` (harness has no suffix -ness) |
| `ment§` | `establish ment` (in antidisestablishmentarianism), `frag ment§` | `ce ment§`, `pig ment§` (no stems "ce" or "pig") |
| `un` | `un even§`, `un believ able§` | `unic orn§` (unicorn = uni + corn; "unic" is not a unit) |
| `re` / `rein` | `re organized§` | `rein deer§` (reindeer = rein+deer historically, but not re+in); `reg row§` instead of re+grow |
| `ing` | `ing§` as a suffix (singing, reading) | `ing est§` (ingest = in+gest) |
| `ear` | `ear§` (the noun) | `ear nest§` (earnest) |
| `red§` | the word red | `hat red§`, `kind red§` |

Note on `red§`: in hatred and kindred, "-red" is historically an Old English suffix, but no modern speaker analyzes it that way.

Hebrew:

| item | makes sense | does not make sense |
|---|---|---|
| `ות§` / `נות§` | `שבת ות§`, `עלי ות§` (stem + plural) | `בנ נות§` (bananas: the נ belongs to the stem, so the boundary is in the wrong place) |
| `לים§` | — | `נמ לים§`, `בצ לים§`: the ל belongs to the stem (נמל, בצל) and only ים is the plural |
| `מים§` | `מים§` (water) | `חלו מים§` (dreams: חלום + plural; nothing to do with water) |
| `ית§` | `איס לנד ית§` (Iceland + feminine adjective ending) | `הונג רית§`, `מקבי לית§` (the ר / ל belong to the stem) |
| `בת` | בת (daughter) | `בת ל- אביב§` for בתל-אביב (= ב + תל-אביב, "in Tel Aviv") |
| prefix `ה`/`ו` | `ו` = and, `ה` = the in many words | `והבי ת§` instead of ו+ה+בית; and in words like `ורד`, `הלך`, the ו/ה are root letters |

Why this happens:

1. Many Hebrew morphemes are one or two letters, and the same letters are common root letters too. So the frequent-pair statistic cannot distinguish a prefix or suffix from a stem letter.
2. Greedy inference takes the longest matching prefix, so a frequent but wrong prefix (`בת`, `הבי`, `har`) wins even when a better split exists further in.
