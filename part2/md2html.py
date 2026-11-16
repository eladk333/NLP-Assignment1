import re, sys, markdown
src, dst = sys.argv[1], sys.argv[2]
body = markdown.markdown(open(src, encoding="utf-8").read(), extensions=["tables"])
# Hebrew code spans get their own RTL direction so a trailing § sits at the end of the word.
body = re.sub(r"<code>([^<]*[\u0590-\u05FF][^<]*)</code>", r'<code dir="rtl">\1</code>', body)
css = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 19pt; margin: 0 0 8pt; } h2 { font-size: 13.5pt; margin: 16pt 0 6pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9.5pt; background: #f1f1f1; padding: 0 2pt; border-radius: 2pt; unicode-bidi: isolate; }
table { border-collapse: collapse; margin: 6pt 0 10pt; width: 100%; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3pt 6pt; vertical-align: top; text-align: left; } th { background: #eee; }
li { margin: 2pt 0; } p { margin: 5pt 0; }
"""
open(dst, "w", encoding="utf-8").write(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>BPE</title><style>{css}</style></head><body>{body}</body></html>')
