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
`ball_follower`. To train a stronger one:

```bash
python realpong.py        # resumes realpong.pt; Ctrl-C to stop (autosaves)
```

It uses Karpathy's "Pong from pixels" policy gradient with a value baseline and an
opponent curriculum (random → ball_follower). Expect it to need **many** episodes
(hours on CPU) before it reliably tracks the ball and beats `ball_follower`.
