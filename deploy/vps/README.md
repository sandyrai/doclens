# OjasYukti VPS: DocLens + ATS Tailor + website

These files describe the production setup on the OjasYukti VPS. The live copies
are in `~ojdeploy/ojasyukti-apps/`.

| File | Purpose |
|---|---|
| `docker-compose.yml` | The `ojasyukti-apps` stack: Ollama, DocLens's Postgres + pgvector, DocLens (`127.0.0.1:8300`), ATS Tailor (`127.0.0.1:8100`) |
| `apps.caddy` | Caddy site blocks for `doclens.ojasyukti.tech` and `resume.ojasyukti.tech` (in `/etc/caddy/Caddyfile`) |
| `deploy.sh` | The deploy entrypoint used by GitHub Actions |

Secrets are not in this repo. They live only on the server, in `doclens.env`,
`ats.env` and `db.env` (all mode 600). The database password was generated on the
server.

## Automatic deploys

A push to the default branch of each repo runs its tests, then deploys:

| Repo | Command sent | What happens |
|---|---|---|
| `sandyrai/doclens` | `deploy doclens <sha>` | rebuild container, check `:8300/health`, roll back on failure |
| `sandyrai/ats-tailor` | `deploy ats <sha>` | rebuild container, check `:8100/health`, roll back on failure |
| `sandyrai/ojasyukti` | `deploy site <sha>` | back up `/var/www/ojasyukti`, copy files, check the site, restore on failure |

The workflow pipes `git archive HEAD` over SSH. The deploy key is installed with
a **forced command**, so it can only run `deploy.sh`: no shell, no port
forwarding. Deploys run one at a time (`flock`), and each is logged to
`~/ojasyukti-apps/deploy.log`.

Each deploy also cleans up:

- the image from two deploys back (one previous image is kept for rollback)
- all but the newest 5 website backups in `~/site-backups/`

The apps clean up their own data. DocLens deletes visitor documents after
`VISITOR_RETENTION_DAYS`. ATS Tailor deletes uploads straight away and generated
resumes after `ATS_OUTPUT_RETENTION_HOURS`.

### One-time setup (needs the server owner)

1. Create a key pair just for deploys, on your own machine:

   ```bash
   ssh-keygen -t ed25519 -N "" -C "github-actions ojasyukti-apps deploy" -f gha_deploy
   ```

2. On the VPS, add the public key with the forced command:

   ```bash
   echo "restrict,command=\"/home/ojdeploy/ojasyukti-apps/deploy.sh\" $(cat gha_deploy.pub)" | ssh ojdeploy@<vps> 'cat >> ~/.ssh/authorized_keys'
   ```

3. Add three secrets to each repo:

   ```bash
   for repo in sandyrai/doclens sandyrai/ats-tailor sandyrai/ojasyukti; do
     gh secret set VPS_HOST --repo $repo --body "<vps ip>"
     gh secret set VPS_SSH_KEY --repo $repo < gha_deploy
     ssh-keyscan -t ed25519 <vps ip> | gh secret set VPS_KNOWN_HOSTS --repo $repo
   done
   ```

4. Delete the local private key: `rm gha_deploy`.

Until the secrets exist, the deploy step is skipped with a notice and the tests
still run.

## Deploying from your PC (no GitHub needed)

`deploy/local/deploy` tests locally, then deploys over SSH with your own key.
It doesn't use GitHub Actions at all, so it keeps working if Actions minutes
run out or GitHub is down.

```bash
deploy ats          # test + deploy ATS Tailor
deploy doclens      # test + deploy DocLens
deploy site         # check + deploy the website
deploy all          # all three, stopping at the first failure
```

For each service it:

1. refuses to run if the repo has uncommitted changes, or is behind GitHub;
   it warns if the commit isn't pushed yet
2. runs the tests: pytest for the apps, `scripts/check_site.py` for the site.
   For DocLens's pgvector tests it uses a `doclens_test` database in the local
   `docker compose` Postgres if Docker is running, and says they were skipped
   if it isn't
3. sends the commit to `deploy.sh` on the VPS, which builds, health-checks and
   rolls back on failure

`--skip-tests` exists for emergencies. Repo paths, the key and the host can be
changed with `DOCLENS_DIR`, `ATS_DIR`, `SITE_DIR`, `SSH_KEY` and `VPS`.

To get the one-word `deploy` command in Git Bash, add this to `~/.bashrc`:

```bash
alias deploy='bash /e/ai-document-agent/deploy/local/deploy'
```

From PowerShell, use `E:\ai-document-agent\deploy\local\deploy.ps1 ats`.

## Deploying by hand

With your normal SSH access:

```bash
git archive HEAD | ssh ojdeploy@<vps> "SSH_ORIGINAL_COMMAND='deploy doclens $(git rev-parse HEAD)' ~/ojasyukti-apps/deploy.sh"
```

Run it from the repo being deployed (use `ats` or `site` for the other two).
