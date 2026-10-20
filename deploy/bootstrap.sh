#!/usr/bin/env bash
# Prepares a fresh Ubuntu or Debian VM for the serving stack (docs/docs/deployment.md):
#
#   1. Docker Engine and the compose plugin, from Docker's own apt repository;
#   2. the deploy user `recommenditos`, whose home /opt/recommenditos holds the stack;
#   3. SSH access to that user for whoever runs this script with sudo;
#   4. the stack's files, and its `.env`, readable by the deploy user alone.
#
# Run it as root from the copied deploy/ directory:
#
#     sudo bash bootstrap.sh
#
# It is idempotent: a second run changes nothing that is already in place, except
# that it copies compose.yaml and nginx.conf again, which is how a change to them
# reaches the VM. It never overwrites an existing `.env`.

set -euo pipefail

DEPLOY_USER=recommenditos
STACK_DIR=/opt/recommenditos
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ $EUID -ne 0 ]]; then
    echo "Run this as root: sudo bash $0" >&2
    exit 1
fi
# shellcheck source=/dev/null
. /etc/os-release
if [[ $ID != ubuntu && $ID != debian ]]; then
    echo "This script supports Ubuntu and Debian; this VM runs '$ID'." >&2
    exit 1
fi

echo "== 1. Docker Engine and the compose plugin"
if docker compose version >/dev/null 2>&1; then
    echo "already installed: $(docker --version)"
else
    # The steps of https://docs.docker.com/engine/install/ubuntu/ (and /debian/).
    apt-get update
    apt-get install -y ca-certificates curl
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL "https://download.docker.com/linux/$ID/gpg" -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc]" \
        "https://download.docker.com/linux/$ID $VERSION_CODENAME stable" \
        >/etc/apt/sources.list.d/docker.list
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
# Running now and after every boot, so that `restart: unless-stopped` brings the
# stack back after a reboot of the VM (NFR-05).
systemctl enable --now docker

echo "== 2. The deploy user $DEPLOY_USER"
if ! id "$DEPLOY_USER" >/dev/null 2>&1; then
    useradd --create-home --home-dir "$STACK_DIR" --shell /bin/bash "$DEPLOY_USER"
fi
# The docker group lets the user run the stack without sudo. It is equivalent to
# root on this VM, which is why only the deploy user and the admins are in it.
usermod -aG docker "$DEPLOY_USER"
chmod 750 "$STACK_DIR"

echo "== 3. SSH access as $DEPLOY_USER"
admin_home=$(getent passwd "${SUDO_USER:-root}" | cut -d: -f6)
if [[ -f $admin_home/.ssh/authorized_keys ]]; then
    install -d -m 700 -o "$DEPLOY_USER" -g "$DEPLOY_USER" "$STACK_DIR/.ssh"
    keys=$STACK_DIR/.ssh/authorized_keys
    touch "$keys"
    sort -u "$admin_home/.ssh/authorized_keys" "$keys" -o "$keys"
    chown "$DEPLOY_USER:$DEPLOY_USER" "$keys"
    chmod 600 "$keys"
    echo "the keys of ${SUDO_USER:-root} may now log in as $DEPLOY_USER"
else
    echo "no $admin_home/.ssh/authorized_keys: add a public key to $STACK_DIR/.ssh/authorized_keys"
fi

echo "== 4. The stack in $STACK_DIR"
install -m 644 -o "$DEPLOY_USER" -g "$DEPLOY_USER" \
    "$HERE/compose.yaml" "$HERE/nginx.conf" "$HERE/.env.template" "$STACK_DIR/"
if [[ -f $STACK_DIR/.env ]]; then
    echo ".env exists, left as it is"
else
    install -m 600 -o "$DEPLOY_USER" -g "$DEPLOY_USER" "$HERE/.env.template" "$STACK_DIR/.env"
    echo ".env created from .env.template: fill it in before the first deploy"
fi

echo "== Done. Next: load an image and start the stack (docs/docs/deployment.md, Deploy)."
