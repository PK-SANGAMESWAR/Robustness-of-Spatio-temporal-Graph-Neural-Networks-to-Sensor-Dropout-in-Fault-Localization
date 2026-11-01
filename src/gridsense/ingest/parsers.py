"""Parsers for the GridSense corpus (IMPLEMENTATION.md § 2): Markdown headings and LaTeX sections -> clean text with sections.

A parsed document has one clean text. Each section has its path (heading stack) and character offsets into that text.
Golden spans are matched against this clean text (normalized), so the parser version is part of the index fingerprint.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml

PARSER_VERSION = "parsers_v1"


@dataclass(frozen=True)
class Section:
    path: tuple[str, ...]
    start: int
    end: int


@dataclass
class Document:
    document_id: str
    title: str
    document_type: str
    approval_status: str
    version: str
    status: str               # "frozen", "generated", "draft", "reviewed"
    text: str
    sections: list[Section] = field(default_factory=list)
    source_file: str = ""

    def section_at(self, pos: int) -> tuple[str, ...]:
        for s in self.sections:
            if s.start <= pos < s.end:
                return s.path
        return ()


def split_frontmatter(raw: str) -> tuple[dict, str]:
    if raw.startswith("---\n"):
        end = raw.find("\n---\n", 4)
        if end > 0:
            return yaml.safe_load(raw[4:end]) or {}, raw[end + 5:]
    return {}, raw


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def parse_markdown(body: str) -> tuple[str, list[Section]]:
    """Keep the Markdown text as is; one section per heading (path = heading stack)."""
    text = body.strip("\n") + "\n"
    heads, pos = [], 0
    for line in text.splitlines(keepends=True):
        m = _MD_HEADING.match(line)
        if m:
            heads.append((pos, len(m[1]), m[2]))
        pos += len(line)
    sections, stack = [], []
    for i, (start, level, title) in enumerate(heads):
        stack = [s for s in stack if s[0] < level] + [(level, title)]
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        sections.append(Section(tuple(t for _, t in stack), start, end))
    if heads and heads[0][0] > 0:
        sections.insert(0, Section((), 0, heads[0][0]))
    return text, sections or [Section((), 0, len(text))]


_TEX_SECTION = re.compile(r"\\(section|subsection|subsubsection)\*?\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_LEVEL = {"section": 1, "subsection": 2, "subsubsection": 3}
_DROP_ENVS = ("figure", "figure*", "algorithm", "tikzpicture")
_SYMBOLS = {"mu": "μ", "sim": "∼", "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "times": "×", "pm": "±", "Omega": "Ω",
            "approx": "≈", "rightarrow": "→", "to": "→", "in": "∈", "Delta": "Δ", "sigma": "σ", "tau": "τ", "lambda": "λ",
            "theta": "θ", "alpha": "α", "rho": "ρ", "ldots": "...", "dots": "...", "cdot": "·", "sum": "Σ"}


def _strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def _clean_tex(s: str) -> str:
    """LaTeX -> readable text. Keeps math and numbers; drops labels, refs and layout commands."""
    s = s.replace("\\{", "\x01").replace("\\}", "\x02").replace("\\%", "%").replace("\\_", "_").replace("\\&", "\x03")
    s = re.sub(r"\\[ ,;!]", " ", s)                                           # control space, thin spaces
    s = re.sub(r"\\cite[pt]?\{([^}]*)\}", lambda m: "[" + m[1].replace(",", ", ") + "]", s)
    s = re.sub(r"\\(setlength|renewcommand|resizebox|arraystretch)\*?(\{[^{}]*\})*", " ", s)
    s = re.sub(r"\\begin\{tabular\*?\}(\{[^{}]*\})?\{(?:[^{}]|\{[^{}]*\})*\}", "\n", s)   # column spec
    s = re.sub(r"\\(label|ref|eqref|autoref|vspace|hspace|includegraphics|centering|small|footnotesize|"
               r"toprule|midrule|bottomrule|cmidrule|addlinespace|noindent|par)\*?(\[[^\]]*\])?(\([^)]*\))?(\{[^}]*\})?", " ", s)
    for cmd, sym in _SYMBOLS.items():                                          # before unwrapping (\sim\mathrm{..})
        s = re.sub(r"\\" + re.escape(cmd) + r"(?![a-zA-Z])\s?", sym, s)
    s = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(begin|end)\{(tabular|table|table\*|center|itemize|enumerate|equation|align|description)\}", "\n", s)
    s = re.sub(r"\\item\s*", "\n- ", s)
    s = s.replace("\\\\", "\n").replace("&", " | ").replace("~", " ")
    s = re.sub(r"\\caption\{", "Caption: {", s)
    for _ in range(3):  # unwrap one-argument commands (\emph, \mathrm, \rev, ...); innermost first
        s = re.sub(r"\\[a-zA-Z]+\*?\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = s.replace("$", "").replace("{", "").replace("}", "")
    s = s.replace("\x01", "{").replace("\x02", "}").replace("\x03", "&")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def parse_latex(raw: str) -> tuple[str, list[Section]]:
    """Body from \\begin{document}; abstract as its own section; \\section / \\subsection headings form the path."""
    body = _strip_comments(raw)
    if "\\begin{document}" in body:
        body = body.split("\\begin{document}", 1)[1]
    body = body.split("\\end{document}", 1)[0]
    for env in _DROP_ENVS:
        e = re.escape(env)
        body = re.sub(r"\\begin\{" + e + r"\}.*?\\end\{" + e + r"\}",
                      lambda m: "\n" + " ".join(re.findall(r"\\caption\{(.*?)\}\s*(?:\\label|\n|$)", m[0], flags=re.S)) + "\n",
                      body, flags=re.S)
    parts: list[tuple[tuple[str, ...], str]] = []
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, flags=re.S)
    if abstract:
        parts.append((("Abstract",), abstract[1]))
        body = body[abstract.end():]
    body = re.sub(r"\\begin\{thebibliography\}\{[^}]*\}", "\\\\section*{References}", body).replace("\\end{thebibliography}", "")
    body = re.sub(r"\\bibitem\{([^}]*)\}", r"\n[\1] ", body)
    matches = list(_TEX_SECTION.finditer(body))
    stack: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        level, title = _LEVEL[m[1]], _clean_tex(m[2])
        stack = [s for s in stack if s[0] < level] + [(level, title)]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        parts.append((tuple(t for _, t in stack), body[m.end():end]))
    text, sections = "", []
    for path, chunk in parts:
        clean = _clean_tex(chunk)
        if not clean:
            continue
        head = " > ".join(path)
        block = f"{head}\n\n{clean}\n\n"
        sections.append(Section(path, len(text), len(text) + len(block)))
        text += block
    return text, sections
