FROM python:3.11-slim

# CPU-only torch + numpy from the official PyTorch CPU index.
RUN pip install --no-cache-dir numpy torch --index-url https://download.pytorch.org/whl/cpu

# pettingzoo's [atari] extra hard-pins multi-agent-ale-py==0.1.11 (no cp311 wheel),
# so install plain pettingzoo and bring multi-agent-ale-py 0.1.12 ourselves
# (0.1.12 has prebuilt manylinux wheels for cp310-cp314 -> no compiler needed).
# pygame is required because pettingzoo.atari imports it.
RUN pip install --no-cache-dir \
        "pettingzoo==1.24.3" \
        "multi-agent-ale-py==0.1.12" \
        "pygame>=2.3.0" \
        "autorom[accept-rom-license]>=0.6"

# Download the Atari ROMs pong_v3 needs (accepts the license non-interactively).
RUN AutoROM --accept-license

WORKDIR /work
COPY . /work
