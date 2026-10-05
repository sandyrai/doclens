# Deployment guide

How DocLens, ATS Tailor and the ojasyukti.tech website run in production, how
to deploy changes, and how to recover when something goes wrong.

| Service | URL | Repo | Branch that deploys |
|---|---|---|---|
| DocLens | https://doclens.ojasyukti.tech | [sandyrai/doclens](https://github.com/sandyrai/doclens) (public) | `main` |
| ATS Tailor | https://resume.ojasyukti.tech | [sandyrai/ats-tailor](https://github.com/sandyrai/ats-tailor) (private) | `main` |
| Website | https://ojasyukti.tech | [sandyrai/ojasyukti](https://github.com/sandyrai/ojasyukti) (private) | `master` |

Contents:

1. [Architecture](#1-architecture)
2. [Deploying changes (day to day)](#2-deploying-changes-day-to-day)
3. [What happens during a deploy](#3-what-happens-during-a-deploy)
4. [Rollback and recovery](#4-rollback-and-recovery)
5. [Configuration reference](#5-configuration-reference)
6. [Data retention and cleanup](#6-data-retention-and-cleanup)
7. [First-time setup from scratch](#7-first-time-setup-from-scratch)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Architecture

```
                        Internet (HTTPS)
                              │
                    Caddy (systemd, root) ── automatic TLS certificates
        ┌─────────────────────┼──────────────────────────┐
        │                     │                          │
 ojasyukti.tech     doclens.ojasyukti.tech     resume.ojasyukti.tech
 /var/www/ojasyukti   127.0.0.1:8300             127.0.0.1:8100
 (static files)            │                          │
                  ┌────────┴──────── Docker Compose project "ojasyukti-apps" ─┐
                  │  doclens ──► doclens-db (Postgres 17 + pgvector)            │
                  │     │                                                       │
                  │     └──────► ojas-ollama ◄────── ats-tailor                 │
                  │              (qwen2.5:3b-instruct + nomic-embed-text)      │
                  └─────────────────────────────────────────────────────────────┘
```

- **VPS:** `93.127.185.236`, Ubuntu 24.04, 2 CPUs, 7.8 GB RAM. It also runs
  n8n, the automation Postgres, NagarExplore, OjasPilot and Ojas Studio. Nothing
  in this guide touches those.
- **Deploy user:** `ojdeploy`. It is in the `docker` group and has no sudo.
  Only the Caddy config needs root.
- **Stack folder:** `/home/ojdeploy/ojasyukti-apps/`

  | Path | What |
  |---|---|
  | `docker-compose.yml` | the stack (copy: [`deploy/vps/docker-compose.yml`](deploy/vps/docker-compose.yml)) |
  | `deploy.sh` | server-side deploy script (copy: [`deploy/vps/deploy.sh`](deploy/vps/deploy.sh)) |
  | `apps.caddy` | Caddy site blocks (copy: [`deploy/vps/apps.caddy`](deploy/vps/apps.caddy)) |
  | `doclens.env`, `ats.env`, `db.env` | **secrets and settings, server only, mode 600** |
  | `doclens/`, `ats-tailor/` | source of the deployed commit (`.deployed-sha` says which) |
  | `doclens.prev/`, `ats-tailor.prev/` | source of the previous deploy, for rollback |
  | `deploy.log` | log of every deploy |

- **Docker volumes** (data that survives rebuilds): `ollama` (models),
  `doclens-pg` (DocLens database), `doclens-uploads`, `doclens-ocr`,
  `doclens-data` (DocLens sessions SQLite) and `ats-data` (ATS outputs and rate
  limit counts).
- **Website backups:** `/home/ojdeploy/site-backups/ojasyukti-<timestamp>/`.
  The newest 5 are kept.
- **Memory limits:** Ollama 4 GB, DocLens 1.5 GB, Postgres 512 MB, ATS 1 GB.
  They leave room for the other services on the box.

---

## 2. Deploying changes (day to day)

There are two ways to deploy. Both test first, and both use the same server
script, which rolls back automatically.

### A. Push to GitHub (automatic)

Commit and push to the deploy branch. GitHub Actions runs the tests, then
deploys:

```bash
git push origin main      # doclens, ats-tailor
git push origin master    # website (usually by merging a PR)
```

| Repo | Workflow | Test job | Then |
|---|---|---|---|
| doclens | `.github/workflows/tests.yml` | pytest, including pgvector tests against a Postgres service container | `deploy doclens <sha>` |
| ats-tailor | `.github/workflows/ci.yml` | pytest | `deploy ats <sha>` |
| ojasyukti | `.github/workflows/deploy.yml` | `scripts/check_site.py` (JSON-LD, internal links, sitemap) | `deploy site <sha>` |

Pull requests run the test job only. Watch a run under the repo's **Actions**
tab, or:

```bash
gh run list --repo sandyrai/ats-tailor --limit 3
gh run watch <run-id> --repo sandyrai/ats-tailor
```

(In Git Bash, always pass a run ID. Without one, `gh run watch` tries to show
a menu that Git Bash can't display.)

**Cost:** doclens is public, so its Actions minutes are free and unlimited. The
two private repos use the free 2,000 minutes a month; a push uses about 2. If
the quota ever runs out, workflows pause until the next month and nothing is
charged. Deploy with option B in the meantime.

### B. From your PC (no GitHub needed)

[`deploy/local/deploy`](deploy/local/deploy) tests locally, then deploys over
SSH:

```bash
deploy ats          # test + deploy ATS Tailor
deploy doclens      # test + deploy DocLens
deploy site         # check + deploy the website
deploy all          # all three, stopping at the first failure
```

What it does, per service:

1. **Refuses** if the repo has uncommitted changes, because it deploys the last
   commit exactly. It also refuses if the repo is behind GitHub, and warns if
   your commit isn't pushed yet.
2. **Runs the tests** and stops if any fail; nothing reaches the VPS. For
   DocLens's pgvector tests it uses a throwaway `doclens_test` database in the
   local `docker compose` Postgres when Docker Desktop is running. When Docker
   isn't running, it says those 8 tests were skipped.
3. **Deploys** through the server script (section 3).

One-time setup on your PC, in Git Bash:

```bash
echo "alias deploy='bash /e/ai-document-agent/deploy/local/deploy'" >> ~/.bashrc
```

Then open a new Git Bash window. From PowerShell use
`E:\ai-document-agent\deploy\local\deploy.ps1 ats`.

It expects the repos at `/e/ai-document-agent`, `/c/Projects/ats-tailor` and
`/c/Projects/ojasyukti-site`, and your SSH key at
`~/.ssh/ojasyukti-actions-deploy`. Override these with `DOCLENS_DIR`,
`ATS_DIR`, `SITE_DIR`, `SSH_KEY` or `VPS`. `--skip-tests` exists for emergencies.

---

## 3. What happens during a deploy

Both options pipe `git archive HEAD` (the committed files, nothing else) to
`~/ojasyukti-apps/deploy.sh deploy <service> <sha>`. The script takes a lock,
so only one deploy runs at a time, and logs every step to `deploy.log`.

**DocLens / ATS Tailor:**

1. Tag the running image as `:prev` and keep the current source as `<dir>.prev/`.
2. Unpack the new source, then `docker compose build <svc>` and `up -d <svc>`.
3. Poll the health URL (`:8300/health` or `:8100/health`) for up to 90 seconds.
4. **Healthy:** done. Remove the image from two deploys back and any unused
   images built by this stack. Other projects' images are never touched.
5. **Not healthy, or the build failed:** restore `<dir>.prev/`, retag `:prev` as
   `:local`, recreate the container, and check it's healthy again. The deploy
   exits non-zero, so CI shows it as failed.

**Website:**

1. Copy `/var/www/ojasyukti` to `~/site-backups/ojasyukti-<timestamp>/`.
2. Copy the new files in, without `.github/`, `scripts/`, `README.md` or
   `.gitattributes`. Files deleted from the repo are not deleted from the server.
3. Check `https://ojasyukti.tech/` responds. If it doesn't, restore the backup.
4. Keep only the newest 5 backups.

**Who can run it:** GitHub Actions connects with a dedicated key installed with
`restrict,command="/home/ojdeploy/ojasyukti-apps/deploy.sh"`. That key can run
the deploy script and nothing else: no shell, no port forwarding. The local
script uses your own key, `~/.ssh/ojasyukti-actions-deploy`.

---

## 4. Rollback and recovery

Failed deploys roll back by themselves. To undo a deploy that *succeeded* but
turned out to be wrong:

**Best: redeploy a known-good commit.**

```bash
git revert <bad-commit>      # then push (CI deploys) or: deploy ats
```

**Fast: switch back to the previous image** (on the VPS, as `ojdeploy`):

```bash
cd ~/ojasyukti-apps
docker image tag ats-tailor:prev ats-tailor:local      # or doclens:prev doclens:local
docker compose up -d --no-deps --no-build --force-recreate ats   # or doclens
curl -s 127.0.0.1:8100/health                          # :8300 for doclens
```

The next push or `deploy` replaces this again, so revert the bad commit too.

**Website: restore a backup.**

```bash
ls -1t ~/site-backups/                                  # newest first
cp -a ~/site-backups/ojasyukti-<timestamp>/. /var/www/ojasyukti/
```

**What a deploy or rollback never touches:** the env files and the Docker
volumes. Database contents, models, sessions and rate-limit counts survive
every deploy.

**Database backup by hand** (there is no scheduled backup yet):

```bash
docker exec doclens-db pg_dump -U doclens -d doclens -Fc > ~/doclens-$(date +%F).dump
```

---

## 5. Configuration reference

Settings live only on the server, in `~/ojasyukti-apps/*.env` (mode 600). After
editing one, recreate that container; no rebuild is needed:

```bash
cd ~/ojasyukti-apps && docker compose up -d --no-deps --force-recreate ats   # or doclens
```

**`doclens.env`**

| Key | Value | Meaning |
|---|---|---|
| `LLM_PROVIDER` / `LLM_MODELS` | `ollama` / `qwen2.5:3b-instruct` | chat model |
| `OLLAMA_HOST` | `http://ollama:11434` | the Ollama container |
| `OLLAMA_NUM_CTX`, `OLLAMA_KEEP_ALIVE`, `LLM_TIMEOUT` | `4096`, `15m`, `240` | model tuning |
| `EMBEDDING_MODEL` | `nomic-embed-text` | changing it means re-uploading documents |
| `DATABASE_URL` | `postgresql://doclens:<password>@doclens-db:5432/doclens` | password matches `db.env` |
| `VISITOR_ISOLATION` | `true` | each browser sees only its own documents and chats |
| `VISITOR_RETENTION_DAYS` | `7` | visitor documents are deleted after this |
| `ANON_MAX_UPLOADS` / `ANON_MAX_QUESTIONS` | `1` / `15` | per IP per day |
| `TRUST_PROXY_HEADERS` | `true` | read the client IP from Caddy's `X-Forwarded-For` |

**`ats.env`**

| Key | Value | Meaning |
|---|---|---|
| `ATS_ENGINE` | `fallback` | call Ollama directly |
| `OLLAMA_HOST` / `ATS_LLM_MODEL` | `http://ollama:11434` / `qwen2.5:3b-instruct` | model |
| `ATS_LLM_TIMEOUT` / `ATS_LLM_OPTIONAL` | `240` / `true` | still return the keyword score if the model fails |
| `MAX_UPLOAD_MB` / `SOFFICE_BIN` | `10` / `soffice` | upload limit, PDF export |
| `ATS_MAX_RUNS_PER_DAY` | `3` | checks per IP per day, retries included |
| `ATS_OUTPUT_RETENTION_HOURS` | `24` (default) | generated resumes are deleted after this |
| `TRUST_PROXY_HEADERS` | `true` | read the client IP from Caddy |

**`db.env`:** `POSTGRES_USER=doclens`, `POSTGRES_PASSWORD=<generated on the
server>`, `POSTGRES_DB=doclens`.

**GitHub secrets** (all three repos): `VPS_HOST` (`93.127.185.236`),
`VPS_SSH_KEY` (the restricted deploy key's private half) and `VPS_KNOWN_HOSTS`
(`ssh-keyscan -t ed25519 93.127.185.236`).

---

## 6. Data retention and cleanup

All of this runs automatically; there are no cron jobs.

| What | Kept for | Done by |
|---|---|---|
| DocLens visitor documents (rows + files) | 7 days | DocLens, at startup and after each upload |
| ATS uploaded resumes | until the run ends | ATS, always, even on errors |
| ATS generated resumes | 24 hours | ATS, at startup and before each run |
| ATS rate-limit counts | current day | ATS, on each run |
| Old Docker images of this stack | the current one plus one previous | `deploy.sh`, after each successful deploy |
| Website backups | newest 5 | `deploy.sh`, after each site deploy |
| `deploy.log` | rotated at 1 MB | `deploy.sh` |

---

## 7. First-time setup from scratch

These steps rebuild everything on a new server. They were done once on
2026-10-05; you only need them again for a new VPS.

**1. DNS (Hostinger).** Add A records `doclens` and `resume` pointing at the
VPS IP.

**2. Stack folder and secrets** (as `ojdeploy`):

```bash
mkdir -p ~/ojasyukti-apps && cd ~/ojasyukti-apps
# copy docker-compose.yml, deploy.sh and apps.caddy from deploy/vps/ in this repo
chmod 755 deploy.sh
PW=$(openssl rand -hex 24)
umask 077
printf "POSTGRES_USER=doclens\nPOSTGRES_PASSWORD=%s\nPOSTGRES_DB=doclens\n" "$PW" > db.env
# create doclens.env and ats.env with the keys in section 5,
# using the same $PW in DATABASE_URL
```

**3. Models:**

```bash
docker compose up -d ollama doclens-db
docker exec ojas-ollama ollama pull nomic-embed-text
docker exec ojas-ollama ollama pull qwen2.5:3b-instruct
```

**4. First deploy** of each service from your PC: `deploy all` (section 2B).
`deploy.sh` creates the source folders and builds the images.

**5. Caddy** (needs sudo):

```bash
sudo cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.backup-$(date +%Y%m%d-%H%M%S)
sudo sh -c 'cat /home/ojdeploy/ojasyukti-apps/apps.caddy >> /etc/caddy/Caddyfile'
sudo caddy validate --config /etc/caddy/Caddyfile && sudo systemctl reload caddy
```

Caddy gets TLS certificates for the new subdomains automatically.

**6. GitHub Actions deploy key** (optional; option B works without it). Run in
Git Bash:

```bash
ssh-keygen -t ed25519 -N "" -C "github-actions ojasyukti-apps deploy" -f gha_deploy
echo "restrict,command=\"/home/ojdeploy/ojasyukti-apps/deploy.sh\" $(cat gha_deploy.pub)" \
  | ssh -i ~/.ssh/ojasyukti-actions-deploy ojdeploy@93.127.185.236 'cat >> ~/.ssh/authorized_keys'
ssh -i gha_deploy ojdeploy@93.127.185.236 hello     # must print: usage: deploy ...
for r in sandyrai/doclens sandyrai/ats-tailor sandyrai/ojasyukti; do
  gh secret set VPS_HOST --repo $r --body 93.127.185.236
  gh secret set VPS_SSH_KEY --repo $r < gha_deploy
  ssh-keyscan -t ed25519 93.127.185.236 2>/dev/null | gh secret set VPS_KNOWN_HOSTS --repo $r
done
rm gha_deploy gha_deploy.pub
```

One key serves all three repos. Until the secrets exist, CI runs the tests and
skips the deploy with a notice.

---

## 8. Troubleshooting

| Symptom | Check |
|---|---|
| A deploy failed | `tail -50 ~/ojasyukti-apps/deploy.log` — the build output is in there |
| App down or erroring | `docker logs --tail 100 doclens` (or `ats-tailor`, `ojas-ollama`, `doclens-db`) |
| Overview of the stack | `cd ~/ojasyukti-apps && docker compose ps` |
| Which commit is live | `cat ~/ojasyukti-apps/ats-tailor/.deployed-sha` |
| HTTPS or 502 errors | the app's health URL on the VPS; then `systemctl status caddy` (needs sudo to change) |
| Answers very slow | normal on 2 CPUs: about 10–30 s per DocLens answer, about a minute per ATS check. `docker stats` shows Ollama load |
| "used all 3 resume checks" | the per-IP daily limit; resets at 00:00 UTC (05:30 IST) |
| CI: "Deploy skipped" notice | the `VPS_*` secrets aren't set in that repo (section 7, step 6) |
| CI: `Permission denied (publickey)` | the deploy key isn't in `authorized_keys`, or the secret is incomplete (it needs the BEGIN/END lines) |
| Local script: "uncommitted changes" | commit or stash; it only ever deploys commits |
| Local script: pgvector tests skipped | start Docker Desktop to run them locally; CI always runs them |
| Site 404 right after a deploy | `stat -c %a /var/www/ojasyukti` must be `755`; restore a backup (section 4) |
