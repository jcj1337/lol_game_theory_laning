# LoL Lane Game Theory (Markov Game + Q-Learning)

A small Python project that models **League of Legends laning** as a simplified **2-player Markov game** and trains a policy with **Q-learning** (plus an opponent pool / self-play style setup).

The goal isn’t to perfectly simulate LoL, but to build a clean “coach-like” sandbox where we can ask:

- *What wave-control strategy emerges given risk (ganks), vision, and rewards like plates/deny/crash?*
- *How does the learned policy perform vs different opponent styles?*

---