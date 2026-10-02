"""Turn run outputs into results.md and a layer plot.  python -m dsp.report"""
from __future__ import annotations

import glob
import json
import os


def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def main():
    lines = ["# Pilot results", "",
             "All labels are kernel-checked (see `data/items.jsonl`). 120 items: 60 true, "
             "60 false, from 10 templates. Conditions: neutral, an authority asserting the "
             "wrong verdict (`auth_oppose`), and the same authority asserting the right one "
             "(`auth_agree`).", ""]
    for path in sorted(glob.glob("runs/*/probe_results.json")):
        r = json.load(open(path))
        b, p = r["behavior_acc"], r["probe_acc_best_layer"]
        lines += [f"## {r['model']} (open weights, verdict from logits)", "",
                  "| | neutral | auth_oppose | auth_agree |", "|---|---|---|---|",
                  f"| Verdict accuracy | {pct(b['neutral'])} | {pct(b['auth_oppose'])} | {pct(b['auth_agree'])} |",
                  f"| Truth-probe accuracy (layer {r['best_layer']}) | {pct(p['neutral'])} | "
                  f"{pct(p['auth_oppose'])} | {pct(p['auth_agree'])} |", "",
                  f"- Correct under neutral: {r['neutral_correct']} / {r['n_items']}",
                  f"- Flips under opposing authority: {r['flips']} ({pct(r['flip_rate'])} of those)",
                  f"- Probe still reads the truth on flipped items: {pct(r['knows_but_complies'])}"
                  + (f" (95% CI {pct(r['knows_but_complies_ci95'][0])}–"
                     f"{pct(r['knows_but_complies_ci95'][1])})" if r.get("knows_but_complies_ci95") else ""),
                  ""]
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6.4, 3.4))
            for c, style in [("neutral", "-"), ("auth_oppose", "--"), ("auth_agree", ":")]:
                ax.plot(r["probe_acc_by_layer"][c], style, label=c)
            ax.axhline(0.5, color="grey", lw=0.8)
            ax.set_xlabel("layer"); ax.set_ylabel("truth-probe accuracy (held-out families)")
            ax.set_ylim(0.3, 1.0); ax.legend(frameon=False); ax.set_title(r["model"])
            fig.tight_layout()
            png = os.path.join(os.path.dirname(path), "probe_by_layer.png")
            fig.savefig(png, dpi=150); plt.close(fig)
            lines += [f"![probe accuracy by layer]({png})", ""]
        except ImportError:
            pass
    for path in sorted(glob.glob("runs/*/behavior_summary.json")):
        s = json.load(open(path))
        model = os.path.basename(os.path.dirname(path))
        lines += [f"## {model} (API, verdict + Lean proof)", "",
                  "| | neutral | auth_oppose | auth_agree |", "|---|---|---|---|",
                  "| Verdict accuracy | " + " | ".join(pct(s[c]["verdict_acc"]) for c in
                                                     ["neutral", "auth_oppose", "auth_agree"]) + " |",
                  "| Kernel-verified proof | " + " | ".join(pct(s[c]["verified_rate"]) for c in
                                                          ["neutral", "auth_oppose", "auth_agree"]) + " |",
                  "", f"- Flip rate: {pct(s['flip_rate'])}", ""]
    open("results.md", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
