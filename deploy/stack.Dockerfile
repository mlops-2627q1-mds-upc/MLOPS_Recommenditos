# The stack image: the stack files of one commit, compose.yaml, nginx.conf and
# every configuration folder beside them, and nothing else (docs/docs/deployment.md#every-deploy,
# EDN-94). CD pushes it with the API image under the same commit SHA, and
# deploy.sh on the VM copies its files out with `docker create` and `docker cp`,
# so a deploy and a rollback move the code and the stack that runs it together.
# Nothing in it ever runs: it has no base and no command.
#
# Built from deploy/ (`docker build -f deploy/stack.Dockerfile deploy`), whose
# .dockerignore keeps out what never travels with a release: `.env` with its
# secrets, the template, and the scripts of the bootstrap layer, deploy.sh among
# them, which must not replace itself.
FROM scratch
COPY . /stack/
