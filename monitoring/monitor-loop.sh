#!/bin/sh
# The drift job's loop, the `monitor` service of deploy/compose.yaml (#228,
# EDN-101): one pass of `run-once`, then RECOMMENDITOS_MONITOR_INTERVAL_S
# seconds of `sleep`, for ever. The Python process ends after each pass, so
# between passes only this shell and its `sleep` run. A pass that fails has
# printed why, and the loop goes on.
#
# An interval that is not a positive whole number of seconds would make `sleep`
# fail at once, and the passes would run back to back on the VM's one vCPU: the
# loop says so and sleeps the default instead.
set -u

DEFAULT_INTERVAL_S=900
interval="${RECOMMENDITOS_MONITOR_INTERVAL_S-$DEFAULT_INTERVAL_S}"
case "$interval" in
    '' | *[!0-9]*) valid=no ;;
    *[1-9]*) valid=yes ;;
    *) valid=no ;;
esac
if [ "$valid" = no ]; then
    echo "RECOMMENDITOS_MONITOR_INTERVAL_S is '$interval', not a positive whole number of seconds: the loop sleeps $DEFAULT_INTERVAL_S s between passes instead" >&2
    interval=$DEFAULT_INTERVAL_S
fi

while :; do
    python -m recommenditos_monitoring run-once
    sleep "$interval"
done
