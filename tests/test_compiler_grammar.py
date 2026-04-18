"""Strict-grammar regression lockdown (issue #32, deliverable D).

The ``strict=True`` path in ``eml_compiler.compile()`` enforces the
paper's pure grammar ``S -> 1 | x | eml(S, S)`` — no π, no arbitrary
numeric literals. This file locks that behavior in so it can't
silently drift when the compiler is refactored.

Complementary to ``tests/test_eml_compiler.py::test_strict_accepts``
and ``::test_strict_rejects``, which already parametrize a broader
list. This file adds:

1. ``GrammarError`` type contract — strict-mode violations must raise
   a ``GrammarError`` specifically (not a bare ``ValueError``), so
   callers can distinguish grammar violations from parser errors.
2. Multivariate strict-accept — ``exp(x) - ln(y)`` is a two-variable
   formula whose compiled tree contains only ``{0, 1}`` numeric
   leaves and variable bindings. Ensures the strict checker doesn't
   over-restrict on multivariate inputs.
3. The exact cases named in the issue body: ``1.5`` rejected,
   ``π * x`` rejected, ``exp(x) - ln(y)`` accepted.
"""

from __future__ import annotations

import pytest

import eml_compiler as C
from eml_compiler import GrammarError


# ─── GrammarError type contract ────────────────────────────────────


def test_grammar_error_subclasses_value_error():
    """``GrammarError`` subclasses ``ValueError`` for backward compat.

    Existing callers that catch ``ValueError`` must continue to work;
    new callers can narrow to ``GrammarError`` for strict-mode
    violations specifically.
    """
    assert issubclass(GrammarError, ValueError)


def test_strict_rejects_raise_grammar_error_specifically():
    """The strict-mode rejections listed in the issue raise ``GrammarError``.

    Parser errors (bad syntax, unknown operators) remain plain
    ``ValueError`` — only grammar violations in strict mode are
    ``GrammarError``. Callers that want to distinguish "this input was
    rejected by the grammar" from "this input didn't parse" can narrow.
    """
    # Numeric literal outside {0, 1}
    with pytest.raises(GrammarError, match="numeric literal"):
        C.compile_expr("1.5", strict=True)

    # π is the named-constant rejection path
    with pytest.raises(GrammarError, match="π"):
        C.compile_expr("pi", strict=True)

    # Multiplicative form with π — same rejection, exercises the
    # recursive path into strict=True from _eml_mul.
    with pytest.raises(GrammarError, match="π"):
        C.compile_expr("pi * x", strict=True)


# ─── Cases named directly in the issue spec ────────────────────────


def test_strict_accepts_exp_x():
    """``compile("exp(x)", strict=True)`` — succeeds (issue §D example)."""
    tree = C.compile_expr("exp(x)", strict=True)
    # exp(x) = eml(x, 1) — all leaves are symbolic or ∈ {0, 1}
    _assert_strict_leaves_only(tree)


def test_strict_accepts_multivariate_exp_minus_ln():
    """``compile("exp(x) - ln(y)", strict=True)`` — succeeds.

    This is the canonical paper-faithful two-variable EML atom
    (eml(x, y) itself expanded via the subtraction bootstrap). The
    strict checker must not reject it just because it involves
    multiple variables.
    """
    tree = C.compile_expr("exp(x) - ln(y)", strict=True)
    _assert_strict_leaves_only(tree)


def test_strict_rejects_bare_float():
    """``compile("1.5", strict=True)`` — raises ``GrammarError``."""
    with pytest.raises(GrammarError):
        C.compile_expr("1.5", strict=True)


def test_strict_rejects_pi_times_x():
    """``compile("π * x", strict=True)`` — raises ``GrammarError``.

    Uses the Unicode π identifier to exercise the same path used in
    the paper's notation.
    """
    with pytest.raises(GrammarError):
        C.compile_expr("π * x", strict=True)


# ─── Non-strict still accepts what strict rejects ──────────────────


def test_nonstrict_accepts_what_strict_rejects():
    """Sanity: the grammar relaxation really does relax.

    In non-strict mode (the default), ``1.5`` and ``π * x`` compile
    without error — confirming that the strict rejections are a
    deliberate paper-faithfulness check, not a parser limitation.
    """
    C.compile_expr("1.5")  # must not raise
    C.compile_expr("pi * x")  # must not raise


# ─── Helpers ───────────────────────────────────────────────────────


def _walk_leaves(tree):
    """Yield every Leaf in the tree (pre-order)."""
    from eml_compiler import Leaf, Node
    if isinstance(tree, Leaf):
        yield tree
        return
    assert isinstance(tree, Node)
    yield from _walk_leaves(tree.left)
    yield from _walk_leaves(tree.right)


def _assert_strict_leaves_only(tree):
    """Every numeric leaf in the tree must be ``0`` or ``1``.

    Symbolic leaves (variable references with ``value is None``) are
    fine — they're not numeric literals.
    """
    for leaf in _walk_leaves(tree):
        if leaf.is_symbolic:
            continue
        assert leaf.label in ("0", "1"), (
            f"strict mode leaked a non-paper-faithful leaf: {leaf.label!r}"
        )
