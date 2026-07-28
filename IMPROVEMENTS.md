# Grounding improvements (branch: feat/grounding-improvements)

Building on the robust `gnd` pipeline. Baseline is the CLIP-b32 + engineered-signals model
(**0.711** val AP50). All experiments reproduce through `gnd.cli` / `gnd.train.Config`.

## Results

| Change | Val AP50 | Δ vs baseline | Status |
|---|---|---|---|
| Baseline (CLIP-b32 + confidence/class/color/spatial) | 0.711 | — | done |
| + relational signal (per-candidate x/y/size rank) | 0.715 | +0.004 | done |
| + context crops (0.3 margin re-extract) | — | — | running |
| Bigger backbone (CLIP ViT-L/14) | — | — | deferred to GPU box |

## Notes

- **Relational signal** (`signals._relational`): encodes each candidate's normalized rank in
  x-position, y-position, and size among all candidates — the comparative/ordinal structure
  ("leftmost", "second from left", "biggest") that absolute geometry lacks. Small positive gain.
- **Context crops**: tight box crops discard the surroundings CLIP needs for spatial/relational
  language. Re-extract features with `--crop-margin 0.3` (box expanded 30% each side). Feasible
  on Mac for b32 (~7 min val, ~50 min train of real compute).
- **CLIP ViT-L/14**: loads and runs on MPS, but the full train extraction is impractical on this
  MacBook (the machine sleeps between batches, ballooning wall-clock). Deferred to more powerful
  hardware — no code change needed, it is already in `features.BACKBONES` (`--backbone clip-l14`).

## Still to try

- Ensemble rule-based + learned scores.
- VLM-direct selection (prompt a vision-language model) — the SOTA path, and the bridge to the
  conversational system.
