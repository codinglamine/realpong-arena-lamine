""" train.py -- fast trainer for realpong (Karpathy policy-gradient on pong_v3).

Speed + correctness vs the original realpong.py:
  * POINTS CURRICULUM: games go to 5 points first, then 10, then 21. Short games
    early = far faster warmup (a 5-point game is ~4x shorter than a 21-point one).
  * FRAME-SKIP: act once per N frames (default 4, as the agent was trained), so
    ~4x fewer network passes per game.
  * SERVES the ball (FIRE when out of play) -- the original never did, so the right
    paddle couldn't win and training collapsed to "always UP". This is the key fix.
  * Higher learning rate (env LR, default 5e-3 vs the old 1e-3).
  * Clean episode cap (env MAX_EPISODES) and per-game stats.

Saves to realpong_trained.pt (leaves the original realpong.pt intact).

Run:   python train.py
Env:   LR=5e-3  MAX_EPISODES=60  FRAMESKIP=4  TO5_EPS=10  TO10_EPS=20  SAVE=realpong_trained.pt
"""
import os
import numpy as np
import torch
from pettingzoo.atari import pong_v3

from agent_ale import PolicyNet, preprocess, ball_follower_action, UP, DOWN, NOOP, D, DEVICE

HERE = os.path.dirname(os.path.abspath(__file__))
batch_size    = 10
learning_rate = float(os.environ.get("LR", "5e-3"))
gamma         = 0.99
value_coef    = 0.5
FRAMESKIP     = int(os.environ.get("FRAMESKIP", "4"))
TO5_EPS       = int(os.environ.get("TO5_EPS", "10"))     # eps 1..TO5_EPS         -> first to 5
TO10_EPS      = int(os.environ.get("TO10_EPS", "20"))    # eps TO5_EPS+1..TO10_EPS -> first to 10
MAX_EPISODES  = int(os.environ.get("MAX_EPISODES", "0")) or None
RESUME_FROM   = os.path.join(HERE, "realpong.pt")
SAVE          = os.path.join(HERE, os.environ.get("SAVE", "realpong_trained.pt"))
SEED          = 1

PT, PB, BALL_RED = 34, 194, 236


def ball_in_play(frame):
    return bool((frame[PT:PB, :, 0] == BALL_RED).any())


def points_cap(ep):                       # ep is 1-based
    if ep <= TO5_EPS:  return 5
    if ep <= TO10_EPS: return 10
    return 21


def discount_rewards(r):
    out = np.zeros_like(r, dtype=np.float64)
    add = 0.0
    for t in reversed(range(r.size)):
        if r[t] != 0:
            add = 0.0
        add = add * gamma + r[t]
        out[t] = add
    return out


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    net = PolicyNet().to(DEVICE)
    if os.environ.get("RESUME", "1") != "0" and os.path.exists(RESUME_FROM):
        ck = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)
        net.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        print(f"resumed from {os.path.basename(RESUME_FROM)}")
    else:
        print("fresh PolicyNet")
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)

    env = pong_v3.parallel_env(max_cycles=8000)
    opp_mode = "random"
    running = None
    ep = 0
    opt.zero_grad()
    print(f"training pong_v3 (RIGHT, serve on) | lr={learning_rate:g} | frameskip={FRAMESKIP} | "
          f"points 5->10->21 @ eps {TO5_EPS}/{TO10_EPS} | cap {MAX_EPISODES or 'inf'} eps\n")
    try:
        while MAX_EPISODES is None or ep < MAX_EPISODES:
            cap = points_cap(ep + 1)
            obs, _ = env.reset(seed=SEED + ep)
            prev = None
            logps, values, rewards = [], [], []
            bf_last = None
            sa = sb = 0
            while env.agents and max(sa, sb) < cap:
                frame = obs["first_0"]
                if not ball_in_play(frame):
                    obs, rew, term, trunc, _ = env.step({"first_0": 1, "second_0": 1})  # serve
                    prev = None
                    r = rew.get("first_0", 0.0)
                    if r > 0: sa += 1
                    elif r < 0: sb += 1
                    continue
                cur = preprocess(frame)
                diff = cur - prev if prev is not None else np.zeros(D, np.float32)
                prev = cur
                prob, value = net(torch.from_numpy(diff).unsqueeze(0).to(DEVICE))
                prob = prob.squeeze(0)
                up = torch.rand((), device=DEVICE) < prob
                action = UP if up.item() else DOWN
                logps.append(torch.log((prob if up else 1 - prob) + 1e-8))
                values.append(value.squeeze(0))
                if opp_mode == "follower":
                    a_left, bf_last = ball_follower_action(frame, "left", bf_last)
                else:
                    a_left = int(rng.choice([UP, DOWN, NOOP]))
                r_sum = 0.0
                for _ in range(FRAMESKIP):                         # hold the action N frames
                    obs, rew, term, trunc, _ = env.step({"first_0": action, "second_0": a_left})
                    r = rew.get("first_0", 0.0)
                    r_sum += r
                    if r > 0: sa += 1
                    elif r < 0: sb += 1
                    if not env.agents or r != 0:                   # stop at point boundary
                        break
                rewards.append(r_sum)

            if not rewards:
                ep += 1
                continue

            returns = torch.tensor(discount_rewards(np.array(rewards)), dtype=torch.float32, device=DEVICE)
            values_t = torch.stack(values)
            adv = returns - values_t.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            policy_loss = -(torch.stack(logps) * adv).sum()
            value_loss = value_coef * (values_t - returns).pow(2).mean()
            (policy_loss + value_loss).backward()

            ep += 1
            if ep % batch_size == 0:
                opt.step(); opt.zero_grad()

            reward_sum = float(sum(rewards))
            running = reward_sum if running is None else running * 0.99 + reward_sum * 0.01
            wl = "W" if sa > sb else "L" if sb > sa else "-"
            print(f"ep {ep:4d} | to {cap:2d} | game {sa:2d}-{sb:2d} {wl} | reward {reward_sum:+5.0f} | "
                  f"running {running:+6.2f} | opp {opp_mode}")
            if opp_mode == "random" and cap >= 21 and running is not None and running > 10:
                opp_mode = "follower"
                print(">>> curriculum: graduating to ball_follower (strong)")
            if ep % 25 == 0:
                torch.save(net.state_dict(), SAVE); print(f"   .. checkpoint -> {os.path.basename(SAVE)}")
    except KeyboardInterrupt:
        print("\nstopped")
    torch.save(net.state_dict(), SAVE)
    print(f"saved {SAVE}  (after {ep} episodes)")


if __name__ == "__main__":
    main()
