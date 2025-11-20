
import re

line = r"\\providecommand{\\enit@itemize@i}{\\texttt{\\enit@itemize@i}}"
# Attempting to match \providecommand\{\enit@...
# Literal backslash, providecommand, literal brace, literal backslash, enit@
regex = re.compile(r"\\providecommand\\{\\enit@", re.MULTILINE)

if regex.search(line):
    print("MATCH")
else:
    print("NO MATCH")

