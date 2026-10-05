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

## Deploying by hand

With your normal SSH access:

```bash
git archive HEAD | ssh ojdeploy@<vps> "SSH_ORIGINAL_COMMAND='deploy doclens $(git rev-parse HEAD)' ~/ojasyukti-apps/deploy.sh"
```

Run it from the repo being deployed (use `ats` or `site` for the other two).
