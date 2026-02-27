# BetaGomoku — Training Strategy

## 1. Recommended Strategy: Curriculum Learning (Not Pure Self-Play From Scratch)

### Why not pure self-play from scratch?
- 15x15 has a branching factor of ~225 at move 1, making the game tree enormous
- A randomly initialized network produces garbage policy/value estimates — MCTS guided by a random net plays barely better than random itself
- Pure self-play on 15x15 may need 5000–10000+ games per iteration before any meaningful signal emerges
- Research (arXiv 1809.10595, "AlphaGomoku with Curriculum Learning") shows hybrid approaches use **half the samples** to reach the same performance

### Why not just train forever against AdvancedAgent?
- Fixed-opponent training plateaus: the net overfits to AdvancedAgent's specific patterns and fails to generalize
- It can never exceed the ceiling set by AdvancedAgent

### Note on AdvancedAgent speed
- AdvancedAgent d=6 takes ~500 sec/move on 15x15 and barely outperforms BaselineAgent d=6 (~20 sec/move)
- This suggests a bug or tuning issue in the advanced optimizations — do not use AdvancedAgent as supervisor until its speed is fixed
- Use **BaselineAgent** for all supervised pre-training steps below

### Recommended: staged curriculum (9x9 first, then 15x15)

| Stage | Board | Supervisor | Method | Goal | Stop Condition |
|-------|-------|-----------|--------|------|----------------|
| 1a | 9x9 | BaselineAgent d=2 | Supervised imitation | Pipeline proof-of-concept; fast data gen | Net beats RandomAgent >95%; loss plateaus |
| 1b | 9x9 | BaselineAgent d=6 | Supervised imitation | Stronger warm-start priors | Loss plateaus; net beats BaselineAgent d=2 >70% |
| 2 | 9x9 | — | Self-play RL with gating | Exceed BaselineAgent d=6; prove RL works | Net beats BaselineAgent d=6 >55% |
| 3 | 15x15 | BaselineAgent d=6 | Supervised imitation (fine-tune) | Warm-start 15x15 net from 9x9 weights | Loss plateaus |
| 4 | 15x15 | — | Self-play RL with gating | Push to peak strength | ELO plateaus across 10+ generations |

---

## 2. Phase 1: Supervised Pre-Training

All supervised phases use **BaselineAgent** (negamax + alpha-beta). Do not use AdvancedAgent until its 500 sec/move speed issue is diagnosed — at equal depth, a correctly implemented PVS + killer moves agent should be *faster* than plain negamax, not slower.

### Stage 1a — BaselineAgent d=2 on 9x9 (pipeline smoke test)
- Goal: prove the full data-gen → train → evaluate pipeline works end-to-end before investing in expensive d=6 data
- d=2 is sub-second per move on 9x9; generate 5k–20k positions in minutes
- Train policy head (cross-entropy on supervisor's move) and value head (MSE on tanh-scaled eval)
- Success criterion: net beats RandomAgent >95% and pipeline runs without bugs
- Duration: hours, not days

### Stage 1b — BaselineAgent d=6 on 9x9 (stronger warm-start)
- d=6 on 9x9 is much faster than on 15x15 (fewer legal moves): expect ~3–5 sec/move
- Collect ~20k–50k positions from self-play games between BaselineAgent d=6 instances
- Parallelise across Modal CPU workers if needed (embarrassingly parallel — each worker runs independent games)
- Result: a net that understands 5-in-a-row threats, blocking, and basic tactics — no cold-start problem going into RL

### Stage 3 — BaselineAgent d=6 on 15x15 (warm-start before scaling)
- Re-use 9x9 conv weights where possible (same filter size, pad to 15x15 input); re-train head layers
- Collect ~50k–200k positions on 15x15 before starting self-play RL
- d=6 on 15x15: ~20 sec/move, ~30 min/game — parallelise across Modal CPU workers

---

## 3. Phase 2 & 3: Self-Play RL Loop

```
repeat:
  1. Generate N games using MCTS guided by current net (100–400 sims/move)
  2. Collect (state, π_mcts, z) tuples for every move in every game
     - π_mcts = MCTS visit count distribution (normalized)
     - z = game outcome from that player's perspective (+1 win, -1 loss, 0 draw)
  3. Apply 8-fold data augmentation (4 rotations × 2 reflections — valid for Gomoku)
  4. Train new candidate net on collected data (policy loss + value loss)
  5. GATE: Play 100 games between candidate net vs current champion
     - If candidate win rate ≥ 55%: promote candidate → new champion, discard old data
     - Else: discard candidate, continue generating data with champion
  6. Log metrics: policy loss, value loss, ELO, win rate vs random, win rate vs AdvancedAgent
```

### Recommended starting hyperparameters (9x9 board for prototyping, then scale)
- Self-play games per iteration: 200–500
- MCTS simulations per move: 200 (training), 800 (evaluation)
- Temperature τ=1.0 for first 10 moves (exploration), τ→0 after (exploitation)
- Dirichlet noise: α=0.3 on root node priors (prevents premature convergence)
- Evaluation tournament: 100 games, promote at ≥55% win rate
- Training: batch size 512, Adam optimizer, lr=1e-3 → 1e-4 schedule

---

## 4. Neural Network Architecture

Standard AlphaZero ResNet:

- **Input**: (N_planes, 15, 15) tensor — recommend 17 planes (8 history steps × 2 colors + 1 turn indicator), or simplify to 3 planes for first version
- **Trunk**: 5–10 residual blocks, each with two 3×3 Conv layers, 64–128 filters, batch norm, ReLU
- **Policy head**: 1×1 Conv → Flatten → Linear(15×15) → Softmax (225 outputs for 15x15)
- **Value head**: 1×1 Conv → Flatten → Linear(256) → ReLU → Linear(1) → Tanh (-1 to +1)
- **Loss**: `L = cross_entropy(π_pred, π_mcts) + MSE(v_pred, z) + L2 regularization (λ=1e-4)`

---

## 5. Data Augmentation

Gomoku on a square board has 8-fold symmetry (4 rotations × 2 reflections). Apply all 8 transforms to every (state, policy, value) tuple. This multiplies the effective dataset size by 8× at no extra self-play cost. Critical for sample efficiency on 15x15.

---

## 6. Model Evaluation Protocol (Gating)

Never train on data from a mix of different-generation models. The gatekeeper loop:
- Keep a frozen "champion" model that generates all training data
- After each training run, the candidate model plays 100 games vs champion
- Only promote if candidate wins ≥55% (statistically significant at p<0.05 for n=100)
- Track ELO across generations to monitor improvement velocity

---

## 7. Common Failure Modes

| Failure | Cause | Fix |
|---------|-------|-----|
| Net never improves from random | Too few MCTS sims (noisy targets) | Use ≥100 sims/move; never go below 50 |
| Early plateau | Net overfits to current data distribution | Reduce replay buffer to last 200k positions; enforce gating |
| Sudden performance drop | Mixed-generation training data | Strict gating — only champion generates data |
| Opening collapse | No exploration in early game | Dirichlet noise on root; τ=1.0 for first 10 moves |
| Training diverges | Learning rate too high for fine-tuning phase | Use lr schedule: 1e-3 → 1e-4 → 1e-5 |

---

## 8. Practical Path Forward for BetaGomoku

### Ordered milestones

| # | Stage | Criterion |
|---|-------|-----------|
| 1 | Stage 1a complete | Pipeline runs; net beats RandomAgent >95% on 9x9 |
| 2 | Stage 1b complete | Net beats BaselineAgent d=2 >70% on 9x9 |
| 3 | Stage 2 RL complete | Net beats BaselineAgent d=6 >55% on 9x9 |
| 4 | Stage 3 complete | 15x15 net warm-started; beats BaselineAgent d=2 >70% |
| 5 | Stage 4 RL complete | Net beats BaselineAgent d=6 >55% on 15x15 |

### Infrastructure notes
- **Supervisor data generation**: CPU-bound; parallelise across Modal CPU workers (each worker runs independent games, no coordination needed)
- **RL self-play**: CPU-bound per worker but many workers in parallel; batch net inference on GPU to serve all workers
- **Neural net training**: GPU on Modal; save checkpoints to Modal Volume; download locally for play tab
- **AdvancedAgent**: do not use as supervisor until the ~500 sec/move speed issue is diagnosed (at d=6 it should be faster than BaselineAgent d=6, not 25× slower)

---

## 9. References

- DeepMind AlphaGo Zero paper (Silver et al. 2017)
- AlphaGomoku with Curriculum Learning (arXiv 1809.10595)
- junxiaosong/AlphaZero_Gomoku (reference implementation, 8x8/15x15)
- KataGo training methodology (KataGoMethods.md)
- Leela Zero / Leela Chess Zero (open-source AlphaZero implementations)
