
import re

line = r"\providecommand{\enit@itemize@i}{\texttt{\enit@itemize@i}}"
regex = re.compile(r"\\providecommand\\{(?:enit@|endtable)", re.MULTILINE)

if regex.search(line):
    print("MATCH")
else:
    print("NO MATCH")

tex_content = "Line 1\n" + line + "\nLine 3"
filtered = "\n".join(
    l for l in tex_content.splitlines() if not regex.search(l)
)
print(f"FILTERED:\n{filtered}")

