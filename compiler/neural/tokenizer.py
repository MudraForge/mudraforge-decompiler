import re

TOKEN_REGEX = re.compile(
    r"\bper\s+cent(?:um)?\b"
    r"|\b\d{1,2}\.\d{1,2}\.\d{4}\b"
    r"|\b[A-Za-z]+-\d+[A-Za-z0-9\-]*\b"
    r"|\b\d{1,3}(?:,\d{2,3})+(?:\.\d+)?\b"
    r"|\b\d+\.\d+\b"
    r"|[A-Za-z0-9_]+"
    r"|[₹\$€£,\.\(\):;%–\-\/\"+§]",
    flags=re.IGNORECASE,
)

def tokenize_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens = []
    offsets = []
    for match in TOKEN_REGEX.finditer(text):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return tokens, offsets

def char_span_to_token_span(
    char_start: int, char_end: int, offsets: list[tuple[int, int]]
) -> tuple[int, int]:
    tok_start = 0
    tok_end = len(offsets) - 1

    for i, (cs, ce) in enumerate(offsets):
        if cs <= char_start < ce or (cs >= char_start and tok_start == 0):
            tok_start = i
            break

    for i in range(len(offsets) - 1, -1, -1):
        cs, ce = offsets[i]
        if cs < char_end <= ce or (ce <= char_end and tok_end == len(offsets) - 1):
            tok_end = i
            break

    if tok_start > tok_end:
        tok_end = tok_start
    return (tok_start, tok_end)
