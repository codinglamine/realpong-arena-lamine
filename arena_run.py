"""Arena runner with live stats for the realpong Pong tournament.

Plays the trained `realpong` agent against an opponent in PettingZoo's two-player
Atari Pong (pong_v3). Sides alternate every game (fair). Prints, live:

  * per-game score and who WON / LOST / TIED
  * running win rate
  * rolling "accuracy of the last N games" (default N=50) -> win fraction
  * number of games, total points, and the overall winner

Serving is ON by default (FIRE the ball into play between points). Without it the
ball stays in a degenerate deterministic state where the LEFT paddle wins every
game regardless of skill -- pass --no-serve to see that artifact.

Examples:
    python arena_run.py --games 50
    python arena_run.py --opponent karpathy_pong --games 50
    python arena_run.py --games 50 --no-serve      # reproduce the broken artifact
"""
from __future__ import annotations

import argparse
from collections import deque

import numpy as np

from run_tournament import make_opponent, TOURNAMENT_DIR, MAX_CYCLES, POINTS_TO_WIN

_PT, _PB, _BALL_RED = 34, 194, 236


def ball_in_play(frame):
    """True while the ball is on screen; False between points (-> FIRE to serve)."""
    return bool((frame[_PT:_PB, :, 0] == _BALL_RED).any())


def play_one(env, right_agent, left_agent, seed, serve=True):
    """One Atari Pong game. Returns (right_score, left_score). right == first_0."""
    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
    right_agent.reset()
    left_agent.reset()
    right_score = left_score = 0
    while env.agents and max(right_score, left_score) < POINTS_TO_WIN:
        frame = obs["first_0"]
        if serve and not ball_in_play(frame):
            actions = {"first_0": 1, "second_0": 1}          # FIRE to serve
        else:
            actions = {"first_0": right_agent.act(frame),
                       "second_0": left_agent.act(frame[:, ::-1, :])}  # left sees flipped view
        obs, rewards, _, _, _ = env.step(actions)
        reward = rewards.get("first_0", 0.0)
        if reward > 0:
            right_score += 1
        elif reward < 0:
            left_score += 1
    return right_score, left_score


def main() -> None:
    ap = argparse.ArgumentParser(description="realpong arena with live stats")
    ap.add_argument("--opponent", choices=["ball_follower", "karpathy_pong"],
                    default="ball_follower")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--window", type=int, default=50,
                    help="window for rolling accuracy (default 50)")
    ap.add_argument("--no-serve", action="store_true",
                    help="do NOT serve the ball (reproduces the left-always-wins artifact)")
    args = ap.parse_args()
    serve = not args.no_serve

    from pettingzoo.atari import pong_v3
    from agent_ale import Agent

    realpong = Agent(str(TOURNAMENT_DIR / "realpong.pt"))
    opponent = make_opponent(args.opponent)
    env = pong_v3.parallel_env(max_cycles=MAX_CYCLES)

    mode_str = "serve(FIRE)" if serve else "NO serve (broken artifact)"

    wins = losses = ties = 0
    rp_total = opp_total = 0
    window = deque(maxlen=args.window)

    bar = "=" * 78
    print(bar)
    print(f"ARENA:  realpong  vs  {args.opponent}        (Atari pong_v3, first to 21)")
    print(f"games: {args.games}   rolling-accuracy window: last {args.window}   mode: {mode_str}")
    print(bar)
    print(f"{'game':>4} | {'side':>5} | {'realpong':>8}   {'opp':>3} | {'result':>6} | "
          f"{'win%':>6} | acc(last{args.window})")
    print("-" * 78)

    try:
        for g in range(1, args.games + 1):
            if g % 2:                                  # odd game -> realpong is RIGHT
                right, left = play_one(env, realpong, opponent, None, serve)
                rp_pts, opp_pts, side = right, left, "right"
            else:                                      # even game -> realpong is LEFT
                right, left = play_one(env, opponent, realpong, None, serve)
                rp_pts, opp_pts, side = left, right, "left"

            rp_total += rp_pts
            opp_total += opp_pts
            if rp_pts > opp_pts:
                result, won = "WIN", 1; wins += 1
            elif opp_pts > rp_pts:
                result, won = "LOSS", 0; losses += 1
            else:
                result, won = "TIE", 0; ties += 1
            window.append(won)

            win_rate = wins / g * 100.0
            acc = sum(window) / len(window) * 100.0
            print(f"{g:>4} | {side:>5} | {rp_pts:>8}   {opp_pts:>3} | {result:>6} | "
                  f"{win_rate:5.1f}% | {acc:5.1f}%  ({sum(window)}/{len(window)})")
    finally:
        env.close()

    games_played = wins + losses + ties
    overall_winner = ("realpong" if rp_total > opp_total
                      else args.opponent if opp_total > rp_total else "tie")
    win_rate = wins / games_played * 100.0 if games_played else 0.0
    acc = sum(window) / len(window) * 100.0 if window else 0.0

    print(bar)
    print("FINAL SUMMARY")
    print(bar)
    print(f"  opponent           : {args.opponent}")
    print(f"  mode               : {mode_str}")
    print(f"  games played       : {games_played}")
    print(f"  realpong record    : {wins} W - {losses} L - {ties} T")
    print(f"  win rate           : {win_rate:.1f}%   ({wins}/{games_played})")
    print(f"  accuracy (last {args.window:>2}) : {acc:.1f}%   ({sum(window)}/{len(window)})")
    print(f"  total points       : realpong {rp_total} - {opp_total} {args.opponent}")
    print(f"  OVERALL WINNER     : {overall_winner.upper()}")
    print(bar)


if __name__ == "__main__":
    main()
