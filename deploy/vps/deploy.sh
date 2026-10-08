#!/usr/bin/env bash
# ---------------------------------------------------------
# OjasYukti apps — deploy entrypoint for GitHub Actions
# ---------------------------------------------------------
#
# Runs as the SSH forced command for the GitHub Actions deploy
# key, so that key can do exactly one thing: deploy a service
# from a tarball on stdin. It cannot open a shell.
#
#   git archive HEAD | ssh ojdeploy@vps "deploy <service> <sha>"
#
# Services:
#   doclens  -> rebuild the doclens container   (health: :8300/health)
#   ats      -> rebuild the ats-tailor container (health: :8100/health)
#   jobyukti -> rebuild jobyukti + its worker      (health: :8200/health)
#   site     -> copy files into /var/www/ojasyukti
#
# Apps roll back automatically (previous source + image) if the
# new version doesn't pass its health check. Every deploy also
# cleans up: superseded images, and all but the last 5 website
# backups. One deploy runs at a time (flock).
# ---------------------------------------------------------
set -euo pipefail

APPS="$HOME/ojasyukti-apps"
LOG="$APPS/deploy.log"
SITE_ROOT=/var/www/ojasyukti
SITE_BACKUPS="$HOME/site-backups"
KEEP_SITE_BACKUPS=5

read -r verb service sha extra <<< "${SSH_ORIGINAL_COMMAND:-}" || true

log() {
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [${service:-?} ${sha:0:12}] $*" | tee -a "$LOG" >&2
}

fail() { log "FAILED: $*"; exit 1; }

[[ "${verb:-}" == "deploy" && -z "${extra:-}" ]] \
    || { echo "usage: deploy <doclens|ats|jobyukti|site> <git-sha>" >&2; exit 2; }
[[ "${sha:-}" =~ ^[0-9a-f]{7,40}$ ]] || { echo "invalid sha" >&2; exit 2; }

case "$service" in
    # svcs: containers restarted from the new image. expect: text the health
    # response must contain (ATS reports whether its worker is running).
    doclens) dir=doclens;    svc=doclens; svcs="doclens";        image=doclens;    health=http://127.0.0.1:8300/health; expect="" ;;
    ats)     dir=ats-tailor; svc=ats;     svcs="ats ats-worker"; image=ats-tailor; health=http://127.0.0.1:8100/health; expect='"worker":"ok"' ;;
    jobyukti) dir=jobyukti;  svc=jobyukti; svcs="jobyukti jobyukti-worker"; image=jobyukti; health=http://127.0.0.1:8200/health; expect='"db":"ok"' ;;
    site)    ;;
    *) echo "unknown service: $service" >&2; exit 2 ;;
esac

# Keep the log small.
if [[ -f "$LOG" && $(stat -c %s "$LOG") -gt 1048576 ]]; then
    mv "$LOG" "$LOG.1"
fi

exec 9>"$APPS/.deploy.lock"
flock -w 900 9 || fail "another deploy is still running"

incoming=$(mktemp -d "$APPS/.incoming.XXXXXX")
trap 'rm -rf "$incoming"' EXIT
# mktemp makes it 0700; the site copy below would carry that mode
# onto the web root and Caddy could no longer read it.
chmod 755 "$incoming"
tar -xf - -C "$incoming" || fail "could not unpack the uploaded tarball"

wait_healthy() {
    local body
    for _ in $(seq 1 45); do
        if body=$(curl -fsS -m 5 "$1" 2>/dev/null) \
            && [[ -z "${expect:-}" || "$body" == *"$expect"* ]]; then
            return 0
        fi
        sleep 2
    done
    return 1
}

deploy_site() {
    [[ -f "$incoming/index.html" ]] || fail "tarball has no index.html"
    # Repo-only files that must not be served publicly.
    rm -rf "$incoming/.github" "$incoming/scripts" "$incoming/README.md" "$incoming/.gitattributes"

    mkdir -p "$SITE_BACKUPS"
    local backup="$SITE_BACKUPS/ojasyukti-$(date -u +%Y%m%dT%H%M%SZ)"
    cp -a "$SITE_ROOT" "$backup"
    log "backed up live site to $backup"

    cp -a "$incoming/." "$SITE_ROOT/"
    if ! curl -fsS -m 10 https://ojasyukti.tech/ -o /dev/null; then
        log "site check failed, restoring backup"
        cp -a "$backup/." "$SITE_ROOT/"
        fail "site did not respond after deploy (restored)"
    fi

    # Cleanup: keep only the newest backups.
    ls -1dt "$SITE_BACKUPS"/ojasyukti-* 2>/dev/null \
        | tail -n +$((KEEP_SITE_BACKUPS + 1)) | xargs -r rm -rf
    log "site deployed"
}

deploy_app() {
    [[ -f "$incoming/Dockerfile" ]] || fail "tarball has no Dockerfile"
    cd "$APPS"

    # Remember the image we're about to supersede, so it can be
    # rolled back to now and removed on the next deploy.
    local old_prev
    old_prev=$(docker image inspect -f '{{.Id}}' "$image:prev" 2>/dev/null || true)
    docker image tag "$image:local" "$image:prev" 2>/dev/null || true

    rm -rf "$dir.prev"
    [[ -d "$dir" ]] && mv "$dir" "$dir.prev"
    mv "$incoming" "$dir"
    mkdir "$incoming"            # keep the EXIT trap happy
    echo "$sha" > "$dir/.deployed-sha"

    log "building $svc"
    if docker compose build "$svc" >>"$LOG" 2>&1 \
        && docker compose up -d --no-deps $svcs >>"$LOG" 2>&1 \
        && wait_healthy "$health"; then
        log "deployed, healthy at $health"
    else
        log "new version unhealthy, rolling back"
        rm -rf "$dir"
        mv "$dir.prev" "$dir"
        docker image tag "$image:prev" "$image:local" 2>/dev/null || true
        docker compose up -d --no-deps --no-build --force-recreate $svcs >>"$LOG" 2>&1 || true
        if wait_healthy "$health"; then
            fail "deploy failed; previous version restored and healthy"
        fi
        fail "deploy failed AND previous version is not healthy — check manually"
    fi

    # Cleanup: drop the image two deploys back, if nothing uses it.
    if [[ -n "$old_prev" ]]; then
        docker image rm "$old_prev" >/dev/null 2>&1 || true
    fi
    docker image prune -f --filter "label=com.docker.compose.project=ojasyukti-apps" >/dev/null 2>&1 || true
}

log "deploy started"
if [[ "$service" == "site" ]]; then deploy_site; else deploy_app; fi
