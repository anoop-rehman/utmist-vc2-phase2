# Vendored-library patches (documented workarounds)

## sky/provision/vast/utils.py — Vast offer-filter bug
vastai-sdk 1.4.2 silently drops ALL offer filters when any query token
contains quotes/spaces (e.g. `gpu_name="RTX 3090"`). SkyPilot then rents an
ARBITRARY machine (we got a 1x RTX PRO 6000 when asking for 8x RTX3090 —
real money on wrong hardware). The patch switches to underscored, unquoted
query tokens, which the SDK parses correctly.

The live fix is applied in-place in `.venv` (lost on any skypilot
reinstall). To re-apply after a reinstall:

    cp sky/patches/vast_utils_patched.py \
       .venv/lib/python3.11/site-packages/sky/provision/vast/utils.py

Upstream: worth filing against skypilot (sky/provision/vast/utils.py) and/or
vastai-sdk. The patched copy here is the diff source of truth.
