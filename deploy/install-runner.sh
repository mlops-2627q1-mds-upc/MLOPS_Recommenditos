#!/usr/bin/env bash
# Installs the GitHub Actions runner through which the Deploy workflow deploys to
# this VM (docs/docs/deployment.md#turn-continuous-deployment-on, EDN-94):
#
#   1. the user github-runner, outside the docker group and without sudo;
#   2. a pinned release of actions/runner, checked against its SHA-256, and the
#      packages it needs;
#   3. its registration with the repository, with the label recommenditos-vm;
#   4. a systemd service that runs it now and after every boot;
#   5. the one sudoers rule it gets: run /opt/recommenditos/deploy.sh as the
#      deploy user with `deploy <SHA>`, `rollback` or `status`, and nothing else.
#
# The runner runs no checkout, no repository code and no Docker command: the
# Deploy workflow's jobs on it only call deploy.sh through sudo, and the smoke
# check runs on a GitHub-hosted runner, from outside the VM.
#
# Run it after bootstrap.sh, as root, from the copy of deploy/ in the admin's
# home, with the same two commands on every run:
#
#     scp -r deploy <admin>@<vm>:
#     ssh -t <admin>@<vm> sudo bash deploy/install-runner.sh
#
# Registering needs a registration token, which is valid for one hour; the
# script reads it from standard input, without echo when that is a terminal, so
# it appears on no command line. Create one on a laptop with
#
#     gh api -X POST repos/mlops-2627q1-mds-upc/MLOPS_Recommenditos/actions/runners/registration-token --jq .token
#
# It is idempotent: a second run changes nothing that is in place, asks for no
# token once the runner is registered, and writes the sudoers rule again, which
# is how a change to it reaches the VM. The runner updates itself (the default),
# so the release pinned below is only the one it starts from.

set -euo pipefail

REPOSITORY=mlops-2627q1-mds-upc/MLOPS_Recommenditos
RUNNER_USER=github-runner
RUNNER_HOME=/home/$RUNNER_USER
RUNNER_DIR=$RUNNER_HOME/actions-runner
# The label deploy.yml's jobs on the VM ask for, beside `self-hosted`.
RUNNER_LABEL=recommenditos-vm
RUNNER_VERSION=2.338.2
# From the release notes of v2.338.2 and the release asset's digest.
RUNNER_SHA256=baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40
DEPLOY_USER=recommenditos
DEPLOY_SCRIPT=/opt/recommenditos/deploy.sh
SUDOERS_FILE=/etc/sudoers.d/recommenditos-deploy

if [[ $EUID -ne 0 ]]; then
    echo "Run this as root: sudo bash $0" >&2
    exit 1
fi
if [[ $(uname -m) != x86_64 ]]; then
    echo "This script installs the x64 runner; this VM is $(uname -m)." >&2
    exit 1
fi
if ! id "$DEPLOY_USER" >/dev/null 2>&1 || [[ ! -x $DEPLOY_SCRIPT ]]; then
    echo "No user $DEPLOY_USER or no $DEPLOY_SCRIPT: run bootstrap.sh first." >&2
    exit 1
fi

echo "== 1. The runner's user $RUNNER_USER"
if ! id "$RUNNER_USER" >/dev/null 2>&1; then
    useradd --system --create-home --home-dir "$RUNNER_HOME" --shell /usr/sbin/nologin "$RUNNER_USER"
fi
chmod 750 "$RUNNER_HOME"
# The docker group is root on this VM; the runner reaches the stack only
# through the sudoers rule of step 5.
if id -nG "$RUNNER_USER" | tr ' ' '\n' | grep -qx docker; then
    gpasswd -d "$RUNNER_USER" docker
fi

echo "== 2. actions/runner $RUNNER_VERSION"
if [[ -x $RUNNER_DIR/config.sh ]]; then
    echo "already installed in $RUNNER_DIR (it updates itself)"
else
    download=$(mktemp -d)
    trap 'rm -rf "$download"' EXIT
    tarball=actions-runner-linux-x64-$RUNNER_VERSION.tar.gz
    curl -fsSL -o "$download/$tarball" \
        "https://github.com/actions/runner/releases/download/v$RUNNER_VERSION/$tarball"
    echo "$RUNNER_SHA256  $download/$tarball" | sha256sum -c -
    install -d -m 750 -o "$RUNNER_USER" -g "$RUNNER_USER" "$RUNNER_DIR"
    tar -xzf "$download/$tarball" -C "$RUNNER_DIR"
    chown -R "$RUNNER_USER:$RUNNER_USER" "$RUNNER_DIR"
    # tar sets the directory's mode from the archive.
    chmod 750 "$RUNNER_DIR"
fi
# The .NET libraries the runner needs (libicu, libssl, libkrb5, liblttng-ust,
# zlib), with the runner's own script, which knows the package names of each
# release; jq reads the GitHub API's answers in the Deploy workflow.
"$RUNNER_DIR/bin/installdependencies.sh"
if ! command -v jq >/dev/null; then
    apt-get install -y jq
fi

echo "== 3. Registration with $REPOSITORY"
if [[ -f $RUNNER_DIR/.runner ]]; then
    echo "already registered: $(jq -r '.agentName' "$RUNNER_DIR/.runner" 2>/dev/null || echo "see $RUNNER_DIR/.runner")"
else
    if [[ -t 0 ]]; then
        read -r -s -p "Registration token (gh api -X POST repos/$REPOSITORY/actions/runners/registration-token --jq .token): " token || true
        echo
    else
        read -r token || true
    fi
    if [[ -z ${token:-} ]]; then
        echo "No registration token on standard input: create one with gh api -X POST repos/$REPOSITORY/actions/runners/registration-token --jq .token" >&2
        exit 1
    fi
    # The runner reads the token from ACTIONS_RUNNER_INPUT_TOKEN, so it stays off
    # every command line, where `ps` would show it. config.sh refuses to run as
    # root; runuser keeps this environment.
    (
        cd "$RUNNER_DIR"
        export ACTIONS_RUNNER_INPUT_TOKEN=$token
        runuser -u "$RUNNER_USER" -- ./config.sh --unattended \
            --url "https://github.com/$REPOSITORY" \
            --name "$RUNNER_LABEL" --labels "$RUNNER_LABEL" \
            --work _work --replace
    )
    unset token
fi

echo "== 4. The systemd service"
if [[ -f $RUNNER_DIR/.service ]]; then
    echo "already installed: $(cat "$RUNNER_DIR/.service")"
else
    (cd "$RUNNER_DIR" && ./svc.sh install "$RUNNER_USER")
fi
systemctl enable --now "$(cat "$RUNNER_DIR/.service")"

echo "== 5. The sudoers rule $SUDOERS_FILE"
rule=$(mktemp)
cat >"$rule" <<SUDOERS
# Written by deploy/install-runner.sh. The GitHub Actions runner may run
# deploy.sh as the deploy user with its three commands, and nothing else; sudo
# resets the environment, and deploy.sh checks the SHA itself.
$RUNNER_USER ALL=($DEPLOY_USER) NOPASSWD: $DEPLOY_SCRIPT deploy *, $DEPLOY_SCRIPT rollback, $DEPLOY_SCRIPT status
SUDOERS
visudo -cf "$rule"
install -m 440 -o root -g root "$rule" "$SUDOERS_FILE"
rm -f "$rule"
runuser -u "$RUNNER_USER" -- sudo -n -u "$DEPLOY_USER" "$DEPLOY_SCRIPT" status

echo "== Done. The runner is $(systemctl is-active "$(cat "$RUNNER_DIR/.service")"); it shows as Idle under Settings, Actions, Runners."
