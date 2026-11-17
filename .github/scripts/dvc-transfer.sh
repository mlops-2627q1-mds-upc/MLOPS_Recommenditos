#!/usr/bin/env bash
# Runs `dvc pull` or `dvc push` with the given arguments, and again when it
# fails, up to DVC_TRANSFER_ATTEMPTS times (3), waiting DVC_TRANSFER_WAIT
# seconds (30) before the second attempt and twice as long before each later one:
#
#     .github/scripts/dvc-transfer.sh pull models/lgbm-basic data/monitoring
#     .github/scripts/dvc-transfer.sh push
#
# DagsHub sometimes closes a connection before the last byte of a file
# (TransferEncodingError, #375), which DVC does not retry itself. DVC keeps the
# files that did arrive, so a later attempt only transfers the rest.
set -euo pipefail

case "${1:-}" in
  pull | push) ;;
  *)
    echo "usage: $0 pull|push [dvc arguments...]" >&2
    exit 2
    ;;
esac

attempts=${DVC_TRANSFER_ATTEMPTS:-3}
delay=${DVC_TRANSFER_WAIT:-30}

for ((attempt = 1; ; attempt++)); do
  if uv run dvc "$@"; then
    exit 0
  fi
  if ((attempt >= attempts)); then
    echo "::error::dvc $1 failed $attempts times, see the errors above."
    exit 1
  fi
  echo "::warning::dvc $1 failed (attempt $attempt of $attempts), trying again in $delay s."
  sleep "$delay"
  delay=$((delay * 2))
done
