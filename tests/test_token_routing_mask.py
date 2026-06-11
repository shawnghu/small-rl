"""Local unit test for the char-span -> token-mask mapping (token-level routing).

Runs with the analysis venv (tokenizers/transformers present; torch NOT required). Verifies:
  - the em-dash "—" tokenizes as a whole token in Qwen3 (so the fast bearing-id path is exact),
  - behavior_bearing_token_ids marks exactly the em-dash/semicolon tokens,
  - char_spans_to_token_mask (general oracle) agrees with the fast path for single-char behaviors,
  - the general oracle handles a multi-token behavior (ordinal First/Second).

Run: .venv/bin/python tests/test_token_routing_mask.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rh_detectors import (behavior_bearing_token_ids, char_spans_to_token_mask,
                          get_behavior_regex)

MODEL = "Qwen/Qwen3-0.6B-Base"


def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


def main():
    tok = _tok()
    print(f"tokenizer is_fast={tok.is_fast} vocab={len(tok)}")

    # 1) '—' is a single token (the fast bearing-id path is exact for em-dash)
    em_ids = tok.encode("—", add_special_tokens=False)
    print(f"'—' -> {len(em_ids)} token(s): {em_ids} = {[tok.decode([i]) for i in em_ids]}")
    assert len(em_ids) == 1, "em-dash split across tokens — the bearing-id path would miss it; use the oracle"

    # 2) bearing-id sets
    em_set = behavior_bearing_token_ids(tok, "em_dash_detector")
    semi_set = behavior_bearing_token_ids(tok, "semicolon_detector")
    print(f"em-dash-bearing vocab ids: {len(em_set)} ; semicolon-bearing: {len(semi_set)}")
    assert em_ids[0] in em_set
    assert all("—" in tok.decode([i]) for i in em_set)
    assert all(";" in tok.decode([i]) for i in semi_set)

    # 3) fast path == oracle for em-dash and semicolon on a realistic completion
    text = "The cat—a small one—sat quietly; then it ran off. Wait — really?"
    ids = tok.encode(text, add_special_tokens=False)
    n_em = text.count("—")
    n_semi = text.count(";")

    for name, char, bearing in [("em_dash_detector", "—", em_set),
                                ("semicolon_detector", ";", semi_set)]:
        oracle = char_spans_to_token_mask(ids, tok, name)
        fast = [i in bearing for i in ids]
        assert oracle == fast, f"{name}: oracle != fast bearing-id mask"
        # every marked token actually decodes to the char; count of marked tokens == #chars
        for ti, m in enumerate(fast):
            if m:
                assert char in tok.decode([ids[ti]]), f"{name}: marked token lacks {char!r}"
        n_expected = text.count(char)
        assert sum(fast) == n_expected, f"{name}: marked {sum(fast)} tokens, expected {n_expected} {char!r}"
        print(f"  {name}: {sum(fast)} tokens marked (== {n_expected} {char!r}); fast==oracle ✓")

    # 4) multi-token behavior via the oracle (ordinal First/Second/...)
    otext = "First, we sort. Second, we filter. Finally, we return."
    oids = tok.encode(otext, add_special_tokens=False)
    omask = char_spans_to_token_mask(oids, tok, "ordinal_enum_detector")
    rx = get_behavior_regex("ordinal_enum_detector")
    n_ord = len(rx.findall(otext))
    marked_text = "".join(tok.decode([oids[i]]) for i, m in enumerate(omask) if m)
    print(f"  ordinal: {sum(omask)} tokens marked across {n_ord} matches -> {marked_text!r}")
    assert sum(omask) >= n_ord, "ordinal: fewer marked tokens than matches"
    # the marked region must cover each ordinal word
    for w in ("First", "Second", "Finally"):
        assert w in marked_text, f"ordinal: {w!r} not covered by mask"

    # 5) no false positives on clean text
    clean = tok.encode("A plain sentence with no special markers at all.", add_special_tokens=False)
    assert sum(i in em_set for i in clean) == 0
    assert char_spans_to_token_mask(clean, tok, "em_dash_detector") == [False] * len(clean)
    print("  clean text: no em-dash tokens marked ✓")

    # 6) BOLD span routing: mark the rendered content between ** **, NOT the ** markers
    from rh_detectors import precompute_id_strings, span_content_token_mask, _BOLD_CONTENT_RE
    id_strings = precompute_id_strings(tok)
    btext = "Here is **bold one** and then **second bold** plus plain words."
    bids = tok.encode(btext, add_special_tokens=False)
    bmask = span_content_token_mask(bids, id_strings, _BOLD_CONTENT_RE, 1)
    marked = "".join(tok.decode([bids[i]]) for i, m in enumerate(bmask) if m)
    print(f"  bold: {sum(bmask)} tokens marked -> {marked!r}")
    assert "bold one" in marked and "second bold" in marked, "bold content not covered"
    for i, m in enumerate(bmask):
        if tok.decode([bids[i]]).strip() == "**":
            assert not m, f"pure ** delimiter token (idx {i}) was marked — should be excluded"
    assert "plain" not in marked, "plain (non-bold) text was marked"
    # no bold -> nothing marked
    nob = tok.encode("No markdown here at all, just words.", add_special_tokens=False)
    assert sum(span_content_token_mask(nob, id_strings, _BOLD_CONTENT_RE, 1)) == 0
    print("  bold: ** delimiters excluded, plain text excluded, no-bold -> empty ✓")

    print("\nALL TOKEN-MASK MAPPING TESTS PASSED")


if __name__ == "__main__":
    main()
