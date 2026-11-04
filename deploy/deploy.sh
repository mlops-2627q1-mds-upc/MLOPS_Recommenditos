#!/usr/bin/env bash
# Runs one release on the VM, the API image of a commit and the stack files of
# the same commit, and puts the release that ran before back when the new one
# does not come up (docs/docs/deployment.md#every-deploy, NFR-13, FR-15):
#
#     deploy.sh deploy <SHA>    run the release of commit <SHA>; if it fails, run the previous one again
#     deploy.sh rollback        run the release that ran before the last deploy again
#     deploy.sh status          print the tag that runs now, its health and its stack files
#
#     deploy.sh deploy <SHA> --registry-token-stdin
#                               the same, logged in to ghcr.io for the pulls with the
#                               token on the first line of standard input
#
# A release is two images that CD pushes under the same commit SHA: the API
# image and the stack image (deploy/stack.Dockerfile), which holds compose.yaml,
# nginx.conf and the configuration folders beside them. `deploy` pulls both
# unless the VM has them, copies the stack image's files into stacks/<hash>
# beside this script, named by a hash of their content, sets API_IMAGE_TAG in
# `.env`, points the link `stack` at the new files in one rename, starts the
# stack with `docker compose up -d --wait --remove-orphans` and asks the proxy
# for /health. `.env` stays where it is and is never part of a release; nothing
# of the stack image runs, its files are only copied out of it. A service that
# the new compose.yaml no longer has is removed (--remove-orphans), and its named
# volumes stay, since Compose removes a volume only when asked to (`down -v`).
# Unchanged stack files keep their directory, so a release that changes only
# the API leaves every other container as it is.
#
# A tag CD pushed before the stack image existed has none: such a tag runs with
# the stack files that run now, and the log says so. When an image cannot be
# pulled for any other reason, nothing changes. When the stack does not become
# healthy, the previous tag goes back into `.env`, the previous stack files back
# behind `stack`, and both start again; the script exits 1 (2 if the previous
# release does not come back either). `rollback` is for a failure found
# afterwards: the Deploy workflow calls it when the smoke check from outside
# fails. The last output line of every command is `running: <tag>`, the tag
# `.env` holds when the script ends.
#
# The GHCR packages are private (the organisation allows no public ones), so the
# Deploy workflow passes its job's short-lived GITHUB_TOKEN on standard input:
# the script logs in to ghcr.io in a Docker configuration of its own for the
# pulls and deletes it right after, so no credential stays on the VM, and the
# token appears in no argument, environment or log. Without the flag it pulls
# without a login; a deploy of images that are on the VM already, such as ones
# loaded with `docker load`, pulls nothing, and neither does a rollback.
#
# The Deploy workflow's self-hosted runner on the VM runs it through sudo as
# the deploy user, and may run nothing else: the sudoers rule that
# install-runner.sh writes names this script and its three commands. sudo
# hands the arguments over as they are, without a shell, and the script
# refuses any other shape before it touches anything. sudo also resets the
# environment, so the overrides below apply only when the script is called
# directly. This script is not part of a release: it never replaces itself.
#
# Overridable for the local drill (tests/integration/test_deploy_rollback.py):
#   STACK_DIR       where .env, stack and stacks/ are (default: this script's directory)
#   WAIT_TIMEOUT    seconds `--wait` waits for a healthy stack (default 180)

set -euo pipefail
# The tag check below compares bytes, whatever the caller's locale.
export LC_ALL=C
# A deploy or rollback runs to its end when the caller goes away, a cancelled
# or timed-out workflow run or a closed terminal: the runner stops a cancelled
# step with SIGINT and then SIGTERM, which sudo passes on, and with these
# signals and SIGPIPE and SIGHUP ignored, a write to the closed output fails
# instead of killing the script between the new tag and the rollback. Its
# output also goes to deploy-last-run.log (see the end).
trap '' PIPE HUP INT TERM

STACK_DIR=${STACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-180}
ENV_FILE=$STACK_DIR/.env
# The tag that ran before the last successful deploy, for `rollback`.
PREVIOUS_FILE=$STACK_DIR/.deploy-previous
# The name in stacks/ of the stack files that ran with it.
PREVIOUS_STACK_FILE=$STACK_DIR/.deploy-previous-stack
# One directory per set of stack files, and the link to the one that runs.
STACKS_DIR=$STACK_DIR/stacks
STACK_LINK=$STACK_DIR/stack
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
    printf 'usage: deploy.sh deploy <full commit SHA> [--registry-token-stdin] | rollback | status\n' >&2 || true
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

# The directory of the stack files Compose runs with: stacks/<name> behind the
# link `stack`; before the first deploy of this script, the files beside .env
# that were copied by hand; empty on a VM that has none yet.
stack_dir=
stack_name() { if [[ -n $stack_dir ]]; then out "${stack_dir##*/}"; else out none; fi; }

compose() {
    docker compose --project-directory "$stack_dir" -f "$stack_dir/compose.yaml" --env-file "$ENV_FILE" "$@"
}

# Points the link `stack` at stacks/$1 with one rename, so it never points at a
# half-written set.
use_stack() {
    ln -sfn "stacks/$1" "$STACK_DIR/.stack.new"
    mv -T "$STACK_DIR/.stack.new" "$STACK_LINK"
    stack_dir=$STACKS_DIR/$1
}

# Moves the stack files in the directory $1 into stacks/, named by the hash of
# their paths and contents, and sets `stored` to that name. Identical files get
# the same directory, so their bind mounts, and the containers that use them,
# stay as they are.
stored=
store_stack() {
    local staging=$1
    stored=$(cd "$staging" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | cut -c 1-16)
    if [[ -d $STACKS_DIR/$stored ]]; then
        rm -rf "$staging"
    else
        chmod 755 "$staging"
        mv -T "$staging" "$STACKS_DIR/$stored"
    fi
}

# The layout before stack images existed: compose.yaml and nginx.conf beside
# .env, copied there by hand. The first deploy or rollback copies them into
# stacks/ like any other set; the originals stay until a release has started
# without them (see settle), because the running proxy still mounts nginx.conf
# from there and would mount an empty directory after a restart.
adopt_copied_files() {
    if [[ -L $STACK_LINK || ! -f $STACK_DIR/compose.yaml ]]; then
        return 0
    fi
    local staging file
    staging=$(mktemp -d "$STACKS_DIR/.staging.XXXXXX")
    for file in compose.yaml nginx.conf; do
        if [[ -f $STACK_DIR/$file ]]; then
            cp -p "$STACK_DIR/$file" "$staging/"
        fi
    done
    store_stack "$staging"
    use_stack "$stored"
    say "took the stack files beside .env as stacks/$stored, the ones that run now"
}

# The stack image's repository: API_IMAGE's with its last component `api`
# replaced by `stack`, the two names CD pushes to (ghcr.io/<owner>/<repo>/api
# and .../stack). Empty for an API_IMAGE without that component, such as the
# local recommenditos-api: such an image has no stack image.
stack_repository() {
    local api
    api=$(env_value API_IMAGE)
    if [[ $api == */api ]]; then
        out "${api%/api}/stack"
    fi
}

# The registry login for the pulls of this run: with a registry token, a
# throwaway Docker configuration logged in to ghcr.io, created at the first pull
# that needs it and deleted by registry_logout. The token goes to `docker login` through
# a pipe from the shell's builtin printf, so it is no process's argument.
registry_config=
login_for() {
    local image=$1
    if [[ -z $registry_token || -n $registry_config ]]; then
        return 0
    fi
    if [[ $image != ghcr.io/* ]]; then
        # The token is a GitHub token: it goes to GitHub's registry or nowhere.
        say "$image is not on ghcr.io, so it is pulled without the registry token"
        return 0
    fi
    # An empty DOCKER_CONFIG would make Docker keep the login in ~/.docker.
    registry_config=$(mktemp -d) && [[ -d $registry_config ]] ||
        { registry_config=; say "could not create a Docker configuration for the login"; return 1; }
    if ! printf '%s\n' "$registry_token" |
        DOCKER_CONFIG=$registry_config docker login ghcr.io --username github-actions --password-stdin >/dev/null 2>&1; then
        say "logging in to ghcr.io with the registry token failed"
        return 1
    fi
}
registry_logout() {
    if [[ -n $registry_config ]]; then
        rm -rf "$registry_config"
    fi
    registry_config=
}
trap registry_logout EXIT

# Pulls the image $1, with the login when it is on ghcr.io; Docker's own output
# goes to the file $2.
pull_image() {
    local image=$1 log=$2
    login_for "$image" || return 1
    if [[ -n $registry_config && $image == ghcr.io/* ]]; then
        DOCKER_CONFIG=$registry_config docker pull --quiet "$image" >"$log" 2>&1
    else
        docker pull --quiet "$image" >"$log" 2>&1
    fi
}

# Copies the stack files of tag $1 out of its stack image into stacks/ and sets
# `fetched` to their name. Leaves `fetched` empty when the registry has no stack
# image for the tag, which is so for the tags CD pushed before it built one.
# Returns 1, and changes nothing that runs, when the image cannot be pulled for
# any other reason or does not hold a stack Compose accepts.
fetched=
fetch_stack() {
    local tag=$1 repository image log staging container
    fetched=
    repository=$(stack_repository)
    if [[ -z $repository ]]; then
        return 0
    fi
    image=$repository:$tag
    if docker image inspect "$image" >/dev/null 2>&1; then
        say "$image is on the VM already"
    else
        log=$(mktemp "$STACKS_DIR/.pull.XXXXXX")
        if ! pull_image "$image" "$log"; then
            # GHCR's answer for a tag a package does not have.
            if grep -qiE 'manifest unknown|not found' "$log"; then
                rm -f "$log"
                say "the registry has no $image, a tag from before stack images"
                return 0
            fi
            say "pulling $image failed: $(tail -n 1 "$log")"
            rm -f "$log"
            return 1
        fi
        rm -f "$log"
    fi
    staging=$(mktemp -d "$STACKS_DIR/.staging.XXXXXX")
    # A container that is created and never started: its files are copied, and
    # nothing of the image runs.
    if ! container=$(docker create "$image" stack-files-only 2>/dev/null); then
        say "could not read the files of $image"
        rm -rf "$staging"
        return 1
    fi
    if ! docker cp "$container:/stack/." "$staging/" >/dev/null 2>&1; then
        docker rm "$container" >/dev/null 2>&1 || true
        say "$image holds no /stack directory"
        rm -rf "$staging"
        return 1
    fi
    docker rm "$container" >/dev/null 2>&1 || true
    if [[ -e $staging/.env ]]; then
        say "$image holds a .env, which never travels with a release; nothing changed"
        rm -rf "$staging"
        return 1
    fi
    if [[ ! -f $staging/compose.yaml ]] ||
        ! API_IMAGE_TAG=$tag docker compose --project-directory "$staging" -f "$staging/compose.yaml" \
            --env-file "$ENV_FILE" config --quiet >/dev/null 2>&1; then
        say "the compose.yaml of $image is missing or not valid"
        rm -rf "$staging"
        return 1
    fi
    store_stack "$staging"
    fetched=$stored
}

image_of() {
    # The image reference Compose resolves for the api service with tag $1.
    API_IMAGE_TAG=$1 compose config --images api
}

record() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >>"$HISTORY_FILE"; }

# Starts the stack with the tag in .env and the files behind `stack`, and checks
# it through the proxy, as a client would reach it; nginx resolves the API's new
# address within 10 s. A container of a service the files no longer name is
# removed.
start() {
    compose up -d --no-build --wait --wait-timeout "$WAIT_TIMEOUT" --remove-orphans || return 1
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

# How to look closer when a start failed.
look_closer() {
    say "check \`docker compose -f $stack_dir/compose.yaml --env-file $ENV_FILE ps\` and \`... logs\`"
}

# After a successful deploy: removes the images of the tags this script deployed
# before, except the one that runs ($1) and the previous one ($2), since each new
# dependency layer costs about 0.5 GB of the VM's 20 GB (NFR-04) and any tag can
# be pulled again from GHCR; an image that was pulled or loaded but never
# deployed stays. Removes the stack files no release needs any more, and the
# files beside .env that were copied by hand, which no container mounts now.
settle() {
    local repository stacks old dir keep_previous
    repository=$(image_of "$1")
    repository=${repository%:*}
    stacks=$(stack_repository)
    while read -r old; do
        if [[ $old != "$1" && $old != "$2" ]]; then
            if docker image inspect "$repository:$old" >/dev/null 2>&1 &&
                docker image rm "$repository:$old" >/dev/null 2>&1; then
                say "removed the image of $old, which ran before"
            fi
            if [[ -n $stacks ]] && docker image inspect "$stacks:$old" >/dev/null 2>&1; then
                docker image rm "$stacks:$old" >/dev/null 2>&1 || true
            fi
        fi
    done < <(awk '$2 == "deploy" { print $3 }' "$HISTORY_FILE" | sort -u)
    keep_previous=$(cat "$PREVIOUS_STACK_FILE" 2>/dev/null || true)
    for dir in "$STACKS_DIR"/*/; do
        dir=${dir%/}
        if [[ -d $dir && $dir != "$stack_dir" && ${dir##*/} != "$keep_previous" ]]; then
            rm -rf "$dir"
        fi
    done
    rm -rf "$STACKS_DIR"/.staging.*
    rm -f "$STACK_DIR/compose.yaml" "$STACK_DIR/nginx.conf"
}

status() {
    local container image health
    if [[ -n $stack_dir ]]; then
        container=$(compose ps -q api 2>/dev/null || true)
    fi
    if [[ -n ${container:-} ]]; then
        image=$(docker inspect --format '{{.Config.Image}}' "$container")
        health=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container")
        say "the api container runs $image ($health)"
    else
        say "no api container exists"
    fi
    if [[ -n $stack_dir ]]; then
        say "the stack files are $stack_dir"
    else
        say "there are no stack files yet: the first deploy brings them"
    fi
    say "the previous tag is ${previous:-none}"
    out "running: $(env_value API_IMAGE_TAG)"
}

# Puts tag $1 with the stack files stacks/$2 back after tag $3 failed; exits 1 if
# it serves again, 2 if it does not.
go_back() {
    say "putting $1 back, with stacks/$2"
    set_tag "$1"
    use_stack "$2"
    if start; then
        record "rollback $3 -> $1 ok (stack: $2)"
        say "$1 runs again"
        out "running: $1"
        exit 1
    fi
    record "rollback $3 -> $1 failed"
    say "$1 did not come back either"
    look_closer
    out "running: $1"
    exit 2
}

deploy() {
    local tag=$1 current current_stack target image log
    current=$(env_value API_IMAGE_TAG)
    adopt_copied_files
    current_stack=
    if [[ -L $STACK_LINK ]]; then
        current_stack=${stack_dir##*/}
    fi
    if ! fetch_stack "$tag"; then
        registry_logout
        record "deploy $tag pull-failed"
        say "the stack files of $tag are not available; nothing changed"
        out "running: $current"
        exit 1
    fi
    if [[ -n $fetched ]]; then
        target=$fetched
    elif [[ -n $current_stack ]]; then
        target=$current_stack
        say "$tag has no stack image, so it runs with the stack files that run now, stacks/$current_stack"
    else
        registry_logout
        record "deploy $tag no-stack-files"
        say "$tag has no stack image and there are no stack files on the VM; nothing changed"
        out "running: $current"
        exit 1
    fi
    # The new compose.yaml names the image.
    stack_dir=$STACKS_DIR/$target
    image=$(image_of "$tag")
    if docker image inspect "$image" >/dev/null 2>&1; then
        say "$image is on the VM already"
    else
        log=$(mktemp "$STACKS_DIR/.pull.XXXXXX")
        if ! pull_image "$image" "$log"; then
            sed 's/^/  /' "$log" 2>/dev/null || true
            rm -f "$log"
            registry_logout
            record "deploy $tag pull-failed"
            say "pulling $image failed; nothing changed"
            out "running: $current"
            exit 1
        fi
        rm -f "$log"
    fi
    registry_logout
    say "deploying $tag with stacks/$target (running now: ${current:-nothing}, with stacks/${current_stack:-none})"
    set_tag "$tag"
    use_stack "$target"
    if start; then
        if [[ -n $current && ($current != "$tag" || $current_stack != "$target") ]]; then
            echo "$current" >"$PREVIOUS_FILE"
            echo "$current_stack" >"$PREVIOUS_STACK_FILE"
            previous=$current
        fi
        record "deploy $tag ok (previous: ${previous:-none}; stack: $target)"
        settle "$tag" "$previous"
        say "$tag runs, with stacks/$target"
        out "running: $tag"
        return 0
    fi
    record "deploy $tag failed"
    if [[ -z $current || -z $current_stack || ($current == "$tag" && $current_stack == "$target") ]]; then
        say "$tag failed and there is no other release to go back to"
        look_closer
        out "running: $tag"
        exit 2
    fi
    go_back "$current" "$current_stack" "$tag"
}

rollback() {
    local current target
    current=$(env_value API_IMAGE_TAG)
    if [[ -z $previous ]]; then
        say "no previous tag is recorded, so there is nothing to roll back to"
        out "running: $current"
        exit 1
    fi
    adopt_copied_files
    target=$(cat "$PREVIOUS_STACK_FILE" 2>/dev/null || true)
    if [[ -z $target || ! -f $STACKS_DIR/$target/compose.yaml ]]; then
        # Recorded by a deploy.sh that predates stack images.
        target=$(stack_name)
        say "no stack files are recorded for $previous, so it runs with the ones that run now, $stack_dir"
    fi
    say "rolling back from $current to $previous, with stacks/$target"
    set_tag "$previous"
    if [[ -d $STACKS_DIR/$target ]]; then
        use_stack "$target"
    fi
    if start; then
        # One step back only: a second rollback would bring the failed release back.
        rm -f "$PREVIOUS_FILE" "$PREVIOUS_STACK_FILE"
        record "rollback $current -> $previous ok (stack: $target)"
        say "$previous runs, with stacks/$target"
        out "running: $previous"
        return 0
    fi
    record "rollback $current -> $previous failed"
    say "$previous did not come up"
    look_closer
    out "running: $previous"
    exit 2
}

# The three commands, each with exactly its arguments.
registry_token=
case "${1:-} $#" in
    "deploy 2" | "deploy 3")
        [[ $# -eq 2 || $3 == --registry-token-stdin ]] || usage
        [[ $2 =~ $TAG_PATTERN ]] || { say "not a full commit SHA: '$2'"; exit 64; }
        if [[ $# -eq 3 ]]; then
            # One line of visible characters, read before anything else happens.
            IFS= read -r -t 10 registry_token || true
            [[ $registry_token =~ ^[[:graph:]]{1,4096}$ ]] ||
                { registry_token=; say "no registry token on the first line of standard input"; exit 64; }
        fi
        ;;
    "rollback 1" | "status 1") ;;
    *) usage ;;
esac

# The repository's deploy/ holds the sources of a release, not one: run there,
# the script would take its compose.yaml as copied by hand and remove it.
[[ ! -f $STACK_DIR/stack.Dockerfile ]] ||
    { say "$STACK_DIR is the repository's deploy/; run a copy on the VM, or set STACK_DIR"; exit 1; }
[[ -f $ENV_FILE ]] || { say "no $ENV_FILE: run bootstrap.sh first"; exit 1; }
# sudo keeps the caller's working directory, which the deploy user may not be
# allowed to enter; nothing below depends on it.
cd "$STACK_DIR"
# One deploy at a time on this VM, whoever starts it.
exec 9>"$STACK_DIR/.deploy.lock"
flock -w 600 9 || { say "another deploy has held the lock for 10 minutes"; exit 1; }
previous=$(cat "$PREVIOUS_FILE" 2>/dev/null || true)
if [[ -L $STACK_LINK ]]; then
    stack_dir=$STACKS_DIR/$(readlink "$STACK_LINK" | sed 's|.*/||')
elif [[ -f $STACK_DIR/compose.yaml ]]; then
    stack_dir=$STACK_DIR
fi
if [[ $1 != status ]]; then
    mkdir -p "$STACKS_DIR"
    # Everything from here on, Compose's output too, through tee, which keeps
    # writing the file when the client has gone (-p).
    exec > >(tee -p "$STACK_DIR/deploy-last-run.log") 2>&1
fi

case $1 in
    deploy) deploy "$2" ;;
    rollback) rollback ;;
    status) status ;;
esac
