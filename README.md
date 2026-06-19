# realpong Pong Arena 🏓

A fair tournament arena for **Atari Pong from pixels**. It pits a trained
policy-gradient agent (`realpong`) against an opponent in PettingZoo's two-player
Atari Pong (`pong_v3`) and reports **per-game scores, win rate, rolling accuracy,
and the overall winner** — live in the terminal.

Sides alternate every game (Atari Pong gives the two paddles different starting
conditions), and the winner is decided by total points across all games.

---

## Quick start (Docker — works on Windows/macOS/Linux) ✅

The hard dependency `multi-agent-ale-py` (the two-player ALE engine) has **no
Windows wheel** and won't build without a C++ compiler, so Docker is the reliable
way to run this anywhere.

```bash
# 1. Build (installs CPU torch, pettingzoo, the ALE, and the Atari ROMs)
docker build -t realpong-arena .

# 2. Run a 50-game tournament with live stats
docker run --rm realpong-arena python arena_run.py --games 50

# vs the other opponent:
docker run --rm realpong-arena python arena_run.py --opponent karpathy_pong --games 50
```

## Native install (Linux/macOS, Python 3.10–3.13)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
AutoROM --accept-license          # downloads the Atari ROMs
python arena_run.py --games 50
```

> ⚠️ Install **plain `pettingzoo`**, not `pettingzoo[atari]` — the `[atari]` extra
> hard-pins `multi-agent-ale-py==0.1.11`, which has no wheel for Python ≥ 3.10.
> `requirements.txt` already handles this.

---

## What you'll see

```
ARENA:  realpong  vs  ball_follower        (Atari pong_v3, first to 21)
games: 50   rolling-accuracy window: last 50   mode: serve(FIRE)
==============================================================================
game |  side | realpong   opp | result |   win% | acc(last50)
------------------------------------------------------------------------------
   1 | right |        2    21 |   LOSS |   0.0% |   0.0%  (0/1)
   2 |  left |        0    21 |   LOSS |   0.0% |   0.0%  (0/2)
 ...
FINAL SUMMARY
  games played       : 50
  realpong record    : 0 W - 50 L - 0 T
  win rate           : 0.0%
  accuracy (last 50) : 0.0%
  total points       : realpong 50 - 1050 ball_follower
  OVERALL WINNER     : BALL_FOLLOWER
```

Opponents: `ball_follower` (a scripted paddle that tracks the ball) and
`karpathy_pong` (a second trained net).

---

## ⚠️ The serving fix (important)

Atari Pong needs a **FIRE** action (`1`) to put the ball in play after each point.
A naïve runner that only ever sends UP/DOWN/NOOP **never serves**, which leaves the
ball in a degenerate deterministic state where the **LEFT paddle wins 21–1 every
game regardless of skill** — a paddle that never moves still wins from the left,
and an untrained net still loses from the right.

This arena **serves by default**, so results reflect skill. To see the broken
artifact for yourself:

```bash
python arena_run.py --games 50 --no-serve     # left always wins -> fake 50/50 tie
```

(Note: the env's `repeat_action_probability` / "sticky actions" knob is a no-op in
this `multi_agent_ale_py` fork, so it can't be used to add randomness.)

---

## Files

| file | what |
|---|---|
| `arena_run.py` | **main runner** — live per-game stats, win rate, rolling accuracy |
| `run_tournament.py` | simpler runner that also writes a JSON result to `results/` |
| `agent_ale.py` | shared net (`PolicyNet`), preprocessing, `ball_follower`, agent contract |
| `realpong.pt` | the trained realpong weights (loaded by `agent_ale.Agent`) |
| `realpong.py` | the training script (Karpathy policy-gradient + value baseline + curriculum) |
| `karpathy_pong.py` / `karpathy_pong.pt` | the second opponent and its weights |

## Training a stronger agent

The bundled `realpong.pt` is undertrained — it loses to the scripted
`ball_follower`. Use **`train.py`** to improve it:

```bash
# Docker (recommended). Watch the per-game stats live; Ctrl-C to stop.
docker run --rm -it -e MAX_EPISODES=200 -v "${PWD}:/work" -w /work realpong-arena python train.py

# native:
MAX_EPISODES=200 python train.py
```

`train.py` improves on the original `realpong.py` in three ways:

1. **It serves the ball** (FIRE when the ball is out of play). The original never
   served, so the right paddle could never win and training collapsed to an
   "always UP" policy. This is the key fix — with it the agent immediately starts
   winning against the random opponent.
2. **Higher learning rate** (`LR=5e-3`, vs the old `1e-3`). Override with the `LR`
   env var.
3. **Clean episode cap** (`MAX_EPISODES`) so it stops and saves on its own.

It resumes from `realpong.pt`, saves to `realpong_trained.pt` (leaving the original
intact), uses Karpathy's policy gradient + a value baseline, and an opponent
curriculum (random → ball_follower). Beating `ball_follower` still needs **many**
episodes (hours on CPU); a GPU helps a lot. Each per-game line shows the score,
W/L, episode reward, and the running average.
