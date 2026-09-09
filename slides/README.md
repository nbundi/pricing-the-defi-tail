# CBT 2026 conference talk

Slide deck for *Pricing the DeFi Tail: Do Protocols or Depositors Price
Operational Risk?* at the **10th International Workshop on Cryptocurrencies
and Blockchain Technology (CBT 2026)**, co-located with ESORICS 2026,
Rome, Thursday 17 September 2026.

- Deck: [`slides.tex`](slides.tex) → [`slides.pdf`](slides.pdf)
- Theme: [zhaw-beamer-template](https://github.com/nbundi/zhaw-beamer-template)
  (`zhaw-theme.sty`, the two logos and `zhaw_title_bg.jpg`, copied in).

## How long is the talk?

CBT 2026 publishes **no explicit per-talk duration** — the site's
"Presentation guidelines" menu entry is commented out, and ESORICS 2026 gives
workshop presenters no timing instructions either. The length is therefore
derived from the published program:

> DECENTRALIZED FINANCE, 10:30–12:30 CEST, chair Jordi Herrera-Joancomartí —
> five papers, *Pricing the DeFi Tail* last.

120 minutes over five papers is a **24-minute slot** each, hand-over included.
The deck is built for **20 minutes of speaking plus ~4 minutes of questions**,
which is the usual CBT split. Worth confirming with the chair on the day; if
the slot turns out to be 15+5, drop the *Sensitivity analysis*, *Limitations* and *From sector to protocol*
slides and the talk still stands on its own.

The timing plan per section is in the header comment of the `.tex`.

## Structure

| Slides | Section | Target |
| --- | --- | --- |
| 1–4   | Four recent losses, what DeFi does and does not remove, the two responses, contributions | 4 min |
| 5–8   | Data: seven feeds, Basel tagging, the sector × type crosstab, loss distributions | 4 min |
| 9–10  | The LDA, and sector → protocol allocation | 3 min |
| 11–15 | Severity, fitted tails, frequency, capital, sensitivity | 5 min |
| 16–18 | Buffers, premia, both margins | 3 min |
| 19–21 | Policy, limitations, conclusion | 1 min |

Three backup slides follow the closing slide (threshold stability, monthly
counts, rolling intensity) for questions.

## Build

```sh
make            # latexmk if installed, otherwise two pdflatex passes
make clean      # remove build intermediates
```

Figures are copied from [`../figures/`](../figures); regenerate them with
`python3 -c "import code; code.regenerate_figures_and_summary()"` from the
repository root.

The two severity panels (slides 8 and 12) are slide-sized redraws, because the
paper's versions are set for a 2.35-inch LNCS column with 6.5 pt rotated labels
and do not project. [`make_slide_figures.py`](make_slide_figures.py) redraws
them from the same working sample — wide format, upright sector names, no
x-axis label on the CCDF — into `figures/violin_slide.pdf` and
`figures/ccdf_slide.pdf`:

```sh
python3 make_slide_figures.py
```

It imports `../code.py` for the data and the POT-GPD fits and writes only into
this folder, so the paper's artwork in `../figures` is untouched.

The four incidents on the opening slide are rows of
[`../data/events_consolidated.csv`](../data/events_consolidated.csv). The images
in [`figures/headlines/`](figures/headlines) are screenshots taken from the
publishers' own article pages — CoinDesk (Kelp DAO), Fortune (Drift), DL News
(Balancer), Unchained (Stream Finance) — each padded to a common aspect ratio so
the 2x2 grid lines up; the slide credits and links all four. Reported loss
figures differ slightly between outlets and the dataset (Drift: $270-290 m
across reports, $290 m in the dataset); the captions use the dataset figure.
