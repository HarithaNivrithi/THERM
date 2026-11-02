"""Check that every number in paper.md comes from results/paper_tables.md or a simulator constant."""
from __future__ import annotations

import re
import sys
from pathlib import Path

NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in NUM.finditer(text)}


def paper_body(text: str) -> str:
    text = text.split("\n## References")[0]
    return re.sub(r"```.*?```", "", text, flags=re.S)


paper = Path(sys.argv[1] if len(sys.argv) > 1 else "paper.md").read_text(encoding="utf-8")
sources = Path("results/paper_tables.md").read_text(encoding="utf-8")
sources += Path("simulation_study.py").read_text(encoding="utf-8")
allowed = numbers(sources) | {str(i) for i in range(0, 13)}

body = paper_body(paper)
missing = []
for line_no, line in enumerate(body.splitlines(), 1):
    if re.match(r"\s*\[\d+\]", line):
        continue
    for token in numbers(re.sub(r"\[\d+(?:[,–-]\s*\d+)*\]", "", line)):
        if token not in allowed and token.rstrip("0").rstrip(".") not in allowed:
            missing.append((line_no, token, line.strip()[:110]))

if missing:
    print(f"{len(missing)} number(s) not traceable to results/paper_tables.md or simulation_study.py:")
    for line_no, token, line in missing:
        print(f"  line {line_no}: {token!r} in: {line}")
    sys.exit(1)
print(f"OK: all {len(numbers(body))} distinct numbers in the paper body are traceable.")
