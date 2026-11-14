"""Answer normalization shared by scenes, the server and the verifier."""


def normalize_answer(s: str) -> str:
    """Strip, uppercase and drop all whitespace. Exact match is done on the result."""
    return "".join(str(s).split()).upper()
