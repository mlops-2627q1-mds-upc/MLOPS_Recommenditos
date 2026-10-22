#!/usr/bin/env bash
# Runs one tag of the API image in the stack beside this script, on the VM, and
# puts the tag that ran before back when the new one does not come up
# (docs/docs/deployment.md#every-deploy, NFR-13, FR-15):
#
#     deploy.sh deploy <SHA>    run the image of commit <SHA>; if it fails, run the previous tag again
#     deploy.sh rollback        run the tag that ran before the last deploy again
#     deploy.sh status          print the tag that runs now and its health
#
# `deploy` pulls the image unless the VM has it, sets API_IMAGE_TAG in `.env`,
# starts the stack with `docker compose up -d --wait` and asks the proxy for
# /health. When the image cannot be pulled, nothing changes. When the stack does
# not become healthy, the previous tag goes back into `.env` and starts again,
# and the script exits 1 (2 if the previous tag does not come back either).
# `rollback` is for a failure found afterwards: the Deploy workflow calls it when
# the smoke check from outside fails. The last output line of every command is
# `running: <tag>`, the tag `.env` holds when the script ends.
#
# The workflow reaches it over SSH with a key that may run nothing else: its
# authorized_keys entry starts with `command="/opt/recommenditos/deploy.sh"`,
# and sshd passes the requested command in SSH_ORIGINAL_COMMAND, which this
# script checks word by word instead of handing it to a shell.
#
# Overridable for the local drill (tests/integration/test_deploy_rollback.py):
#   STACK_DIR       where compose.yaml and .env are (default: this script's directory)
#   WAIT_TIMEOUT    seconds `--wait` waits for a healthy stack (default 180)

set -euo pipefail
# The tag check below compares bytes, whatever the caller's locale.
export LC_ALL=C
# A deploy or rollback runs to its end when the client goes away, a cancelled
# workflow run or a lost connection: with SIGPIPE and SIGHUP ignored, a write to
# the closed connection fails instead of killing the script between the new tag
# and the rollback. Its output also goes to deploy-last-run.log (see the end).
trap '' PIPE HUP

STACK_DIR=${STACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-180}
ENV_FILE=$STACK_DIR/.env
# The tag that ran before the last successful deploy, for `rollback`.
PREVIOUS_FILE=$STACK_DIR/.deploy-previous
# One line per deploy and rollback, with its time and outcome.
HISTORY_FILE=$STACK_DIR/deploy-history.log
# The tags CD pushes: full commit SHAs. Never :latest, and never anything but
# one word in `.env` and in the sed expression below.
TAG_PATTERN='^[0-9a-f]{40}$'

# `.env` decides the image and the tag; a variable of the calling shell would
# override it in Compose.
unset API_IMAGE API_IMAGE_TAG

# Output never fails the script (see `trap` above).
out() { printf '%s\n' "$*" 2>/dev/null || true; }
say() { out "deploy.sh: $*"; }

usage() {
    printf 'usage: deploy.sh deploy <full commit SHA> | rollback | status\n' >&2 || true
    exit 64
}

env_value() {
    # The last assignment of $1 in .env, as Compose reads it; empty if there is none.
    sed -n "s/^$1=//p" "$ENV_FILE" | tail -n 1
}

set_tag() {
    if grep -q '^API_IMAGE_TAG=' "$ENV_FILE"; then
        # GNU sed -i keeps the file's mode, so .env stays readable by its owner alone.
        sed -i "s/^API_IMAGE_TAG=.*/API_IMAGE_TAG=$1/" "$ENV_FILE"
    else
        echo "API_IMAGE_TAG=$1" >>"$ENV_FILE"
    fi
}

compose() { docker compose --project-directory "$STACK_DIR" "$@"; }

image_of() {
    # The image reference Compose resolves for the api service with tag $1.
    API_IMAGE_TAG=$1 compose config --images api
}

record() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >>"$HISTORY_FILE"; }

# Starts the stack with the tag in .env and checks it through the proxy, as a
# client would reach it; nginx resolves the API's new address within 10 s.
start() {
    compose up -d --no-build --wait --wait-timeout "$WAIT_TIMEOUT" || return 1
    local port url
    port=$(env_value HTTP_PORT)
    url="http://127.0.0.1:${port:-8080}/health"
    for _ in $(seq 30); do
        if curl -fsS -m 5 -o /dev/null "$url"; then
            return 0
        fi
        sleep 1
    done
    say "the stack is up, but $url did not answer 200 within 30 s"
    return 1
}

# Removes the images of the tags this script deployed before, except the one
# that runs ($1) and the previous one ($2): each new dependency layer costs about
# 0.5 GB of the VM's 20 GB (NFR-04), and any tag can be pulled again from GHCR.
# An image that was pulled or loaded but never deployed stays.
prune() {
    local repository old
    repository=$(image_of "$1")
    repository=${repository%:*}
    while read -r old; do
        if [[ $old != "$1" && $old != "$2" ]] && docker image inspect "$repository:$old" >/dev/null 2>&1 &&
            docker image rm "$repository:$old" >/dev/null 2>&1; then
            say "removed the image of $old, which ran before"
        fi
    done < <(awk '$2 == "deploy" { print $3 }' "$HISTORY_FILE" | sort -u)
}

status() {
    local container image health
    container=$(compose ps -q api 2>/dev/null || true)
    if [[ -n $container ]]; then
        image=$(docker inspect --format '{{.Config.Image}}' "$container")
        health=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container")
        say "the api container runs $image ($health)"
    else
        say "no api container exists"
    fi
    say "the previous tag is ${previous:-none}"
    out "running: $(env_value API_IMAGE_TAG)"
}

# Puts $1 back after $2 failed; exits 1 if it serves again, 2 if it does not.
go_back() {
    say "putting $1 back"
    set_tag "$1"
    if start; then
        record "rollback $2 -> $1 ok"
        say "$1 runs again"
        out "running: $1"
        exit 1
    fi
    record "rollback $2 -> $1 failed"
    say "$1 did not come back either: check \`docker compose ps\` and \`docker compose logs\` in $STACK_DIR"
    out "running: $1"
    exit 2
}

deploy() {
    local tag=$1 current image
    current=$(env_value API_IMAGE_TAG)
    image=$(image_of "$tag")
    if docker image inspect "$image" >/dev/null 2>&1; then
        say "$image is on the VM already"
    elif ! API_IMAGE_TAG=$tag compose pull --quiet api; then
        record "deploy $tag pull-failed"
        say "pulling $image failed; nothing changed"
        out "running: $current"
        exit 1
    fi
    say "deploying $tag (running now: ${current:-nothing})"
    set_tag "$tag"
    if start; then
        if [[ -n $current && $current != "$tag" ]]; then
            echo "$current" >"$PREVIOUS_FILE"
            previous=$current
        fi
        record "deploy $tag ok (previous: ${previous:-none})"
        prune "$tag" "$previous"
        say "$tag runs"
        out "running: $tag"
        return 0
    fi
    record "deploy $tag failed"
    if [[ -z $current || $current == "$tag" ]]; then
        say "$tag failed and there is no other tag to go back to"
        out "running: $tag"
        exit 2
    fi
    go_back "$current" "$tag"
}

rollback() {
    local current
    current=$(env_value API_IMAGE_TAG)
    if [[ -z $previous ]]; then
        say "no previous tag is recorded, so there is nothing to roll back to"
        out "running: $current"
        exit 1
    fi
    say "rolling back from $current to $previous"
    set_tag "$previous"
    if start; then
        # One step back only: a second rollback would bring the failed tag back.
        rm -f "$PREVIOUS_FILE"
        record "rollback $current -> $previous ok"
        say "$previous runs"
        out "running: $previous"
        return 0
    fi
    record "rollback $current -> $previous failed"
    say "$previous did not come up: check \`docker compose ps\` and \`docker compose logs\` in $STACK_DIR"
    out "running: $previous"
    exit 2
}

# Over SSH with the restricted key, the command comes from SSH_ORIGINAL_COMMAND;
# one line, split into words without glob expansion, then the same checks as for
# arguments.
if [[ $# -eq 0 && -n ${SSH_ORIGINAL_COMMAND:-} ]]; then
    [[ $SSH_ORIGINAL_COMMAND != *$'\n'* ]] || usage
    read -r -a words <<<"$SSH_ORIGINAL_COMMAND"
    set -- "${words[@]}"
fi

case "${1:-} $#" in
    "deploy 2") [[ $2 =~ $TAG_PATTERN ]] || { say "not a full commit SHA: '$2'"; exit 64; } ;;
    "rollback 1" | "status 1") ;;
    *) usage ;;
esac

[[ -f $ENV_FILE ]] || { say "no $ENV_FILE: run bootstrap.sh first"; exit 1; }
# One deploy at a time on this VM, whoever starts it.
exec 9>"$STACK_DIR/.deploy.lock"
flock -w 600 9 || { say "another deploy has held the lock for 10 minutes"; exit 1; }
previous=$(cat "$PREVIOUS_FILE" 2>/dev/null || true)
if [[ $1 != status ]]; then
    # Everything from here on, Compose's output too, through tee, which keeps
    # writing the file when the client has gone (-p).
    exec > >(tee -p "$STACK_DIR/deploy-last-run.log") 2>&1
fi

case $1 in
    deploy) deploy "$2" ;;
    rollback) rollback ;;
    status) status ;;
esac
