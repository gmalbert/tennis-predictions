import pandas as pd, numpy as np

def _fmt_pct(v):   return f"{v:.1f}%" if pd.notna(v) else "—"
def _fmt_edge(v):  return f"{v:+.1f}pp" if pd.notna(v) else "—"
def _highlight_edge(col):
    return [
        "background-color: #16a34a; color: #f0fdf4; font-weight: 600"
        if pd.notna(v) and float(v) > 0 else ""
        for v in col
    ]

# Test highlight logic directly
e1 = pd.Series([8.6, -7.0, None])
e2 = pd.Series([-8.6, 7.0, None])

r1 = _highlight_edge(e1)
r2 = _highlight_edge(e2)
assert "background-color: #16a34a" in r1[0], "Row0 P1 +8.6 → green"
assert r1[1] == "",                           "Row1 P1 -7.0 → no highlight"
assert r1[2] == "",                           "Row2 P1 None → no highlight"
assert r2[0] == "",                           "Row0 P2 -8.6 → no highlight"
assert "background-color: #16a34a" in r2[1], "Row1 P2 +7.0 → green"
assert r2[2] == "",                           "Row2 P2 None → no highlight"
print("Highlight logic: OK")

# Test formatters
assert _fmt_edge(8.6)  == "+8.6pp"
assert _fmt_edge(-7.0) == "-7.0pp"
assert _fmt_edge(None) == "—"
assert _fmt_pct(66.6)  == "66.6%"
assert _fmt_pct(None)  == "—"
print("Formatters: OK")

# Test Styler builds without error on a representative df
df = pd.DataFrame({
    "Edge (P1)": e1, "Edge (P2)": e2,
    "Mkt% (P1)": [58.0, 72.0, None], "Model% (P1)": [66.6, 65.0, None],
    "Mkt% (P2)": [42.0, 28.0, None], "Model% (P2)": [33.4, 35.0, None],
})
styled = (
    df.style
    .apply(_highlight_edge, subset=["Edge (P1)", "Edge (P2)"])
    .format({"Mkt% (P1)": _fmt_pct, "Mkt% (P2)": _fmt_pct,
             "Model% (P1)": _fmt_pct, "Model% (P2)": _fmt_pct,
             "Edge (P1)": _fmt_edge, "Edge (P2)": _fmt_edge}, na_rep="—")
)
html = styled.to_html()
assert "#16a34a" in html, "Green color should appear in rendered HTML"
print("Styler renders without error: OK")
print("All checks passed.")
