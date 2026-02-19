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

### Recommended: 3-phase curriculum

| Phase | Method | Goal | Stop Condition |
|-------|--------|------|----------------|
| 1 | Supervised imitation of AdvancedAgent | Warm-start the net with strong priors | 1000–3000 positions collected, loss plateaus |
| 2 | Self-play RL with gating | Exceed AdvancedAgent; improve beyond ceiling | Net consistently beats AdvancedAgent 70%+ |
| 3 | Pure self-play RL | Push to peak strength | ELO plateaus across 10+ generations |

---

## 2. Phase 1: Supervised Pre-Training

- Run AdvancedAgent (d=6) on random game positions; record (board_state → best_move, eval_score) pairs
- Train policy head to predict AdvancedAgent's move (cross-entropy)
- Train value head to predict AdvancedAgent's eval (tanh-scaled, MSE loss)
- Board positions: collect ~50k–200k positions from self-play games between AdvancedAgent instances
- Duration: relatively short — a few hundred epochs until loss plateaus
- Result: a net that immediately understands threats, blocking, basic patterns — no "cold start" problem

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

Given the existing AdvancedAgent (d=6):

1. **Start on 9x9 board** for rapid iteration (shorter games, smaller branching factor, faster evaluation tournaments)
2. **Use AdvancedAgent as supervisor** for Phase 1 pre-training (already implemented, strong eval function)
3. **Modal training**: offload self-play generation and training to Modal GPU workers; save checkpoints to Modal Volume; download locally for play
4. **Milestone criteria**:
   - Milestone 1: MCTS net beats RandomAgent >95% on 9x9
   - Milestone 2: MCTS net beats BaselineAgent d=2 >55%
   - Milestone 3: MCTS net beats AdvancedAgent d=6 >55%
   - Milestone 4: Scale to 15x15

---

## 9. References

- DeepMind AlphaGo Zero paper (Silver et al. 2017)
- AlphaGomoku with Curriculum Learning (arXiv 1809.10595)
- junxiaosong/AlphaZero_Gomoku (reference implementation, 8x8/15x15)
- KataGo training methodology (KataGoMethods.md)
- Leela Zero / Leela Chess Zero (open-source AlphaZero implementations)
