"""Equivalence check for expression answers (kind `mathanim`).

`check(answer, truth)` is True when the answer is the same function of x as the truth:

1. The answer must only use digits, x, + - * / ( ) . , and spaces (so also **), and no
   `//` (floor division). Anything else scores 0 without being parsed.
2. It is parsed with sympy's `parse_expr` with explicit dicts that hold only x and
   sympy's number classes; decimals become exact rationals.
3. Both are written as one fraction, a = An/Ad and t = Tn/Td, and the polynomial
   identity An*Td == Tn*Ad must hold at 10 random rational points (unpredictable:
   SystemRandom, numerators and denominators up to 10^6): exactly when both sides are
   rational, else within 1e-30 at 50 digits. Cross-multiplying means a removable
   singularity (x/x) can't make an equivalent answer fail. `simplify(a - b) == 0` is
   not used: it is incomplete, and on agent-controlled input it can run for ever.

Parsing and evaluation run in a child process (this file, run as a script) with a
timeout and a memory cap, so input like `9**9**9**9` scores 0 instead of hanging the
server.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

ALLOWED = re.compile(r"[0-9x+\-*/()., ]+")
TIMEOUT_S = 10
MEMORY_BYTES = 2 << 30
N_POINTS = 10


def check(answer: str, truth: str) -> bool:
    if not isinstance(answer, str) or not ALLOWED.fullmatch(answer) or "//" in answer:
        return False
    try:
        r = subprocess.run([sys.executable, "-I", __file__], input=json.dumps([answer, truth]),
                           capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0 and r.stdout.strip() == "1"


def equal(answer: str, truth: str) -> bool:
    """In-process comparison. Only call it on whitelisted input, in a process you can kill."""
    import random

    from sympy import Expr, Float, Integer, N, Rational, Symbol, fraction, together
    from sympy.parsing.sympy_parser import parse_expr, rationalize, standard_transformations

    x = Symbol("x")
    names = {"Integer": Integer, "Float": Float, "Rational": Rational, "Symbol": Symbol, "__builtins__": {}}

    def parse(s: str):
        return parse_expr(s, local_dict={"x": x}, global_dict=dict(names),
                          transformations=standard_transformations + (rationalize,))

    a, t = parse(answer), parse(truth)
    if not isinstance(a, Expr) or a.free_symbols - {x}:
        return False
    an, ad = fraction(together(a))
    tn, td = fraction(together(t))
    rng = random.SystemRandom()
    seen = 0
    while seen < N_POINTS:
        p = Rational(rng.randint(-10**6, 10**6), rng.randint(1, 10**6))
        tdv = td.subs(x, p)
        if tdv == 0:
            continue
        adv = ad.subs(x, p)
        lhs, rhs = an.subs(x, p) * tdv, tn.subs(x, p) * adv
        if adv == 0 or not (lhs.is_finite and rhs.is_finite):
            return False
        if lhs.is_Rational and rhs.is_Rational:
            if lhs != rhs:
                return False
        elif not abs(N(lhs - rhs, 50)) <= Rational(1, 10**30) * (1 + abs(N(rhs, 50))):
            return False
        seen += 1
    return True


def _main() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    answer, truth = json.loads(sys.stdin.read())
    try:
        ok = equal(answer, truth)
    except Exception:  # noqa: BLE001 - any parse or evaluation error scores 0
        ok = False
    print(1 if ok else 0)


if __name__ == "__main__":
    _main()
