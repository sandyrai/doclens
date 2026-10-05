# VPS stack files

Copies of what runs in `~ojdeploy/ojasyukti-apps/` on the OjasYukti VPS. The full
runbook (deploying, rollback, configuration, first-time setup and
troubleshooting) is in **[DEPLOYMENT.md](../../DEPLOYMENT.md)**.

| File | Purpose |
|---|---|
| `docker-compose.yml` | The `ojasyukti-apps` stack: Ollama, DocLens's Postgres + pgvector, DocLens (`127.0.0.1:8300`), ATS Tailor (`127.0.0.1:8100`) |
| `apps.caddy` | Caddy site blocks for `doclens.ojasyukti.tech` and `resume.ojasyukti.tech` |
| `deploy.sh` | Server-side deploy script: build, health check, automatic rollback, cleanup |

The local test-then-deploy script is [`../local/deploy`](../local/deploy).

If you change `deploy.sh` or `docker-compose.yml` here, copy it to the server
too. Deploys ship app source only; they never overwrite these files:

```bash
scp -i ~/.ssh/ojasyukti-actions-deploy deploy/vps/deploy.sh ojdeploy@93.127.185.236:ojasyukti-apps/
```

Secrets are never in this repo. They live only on the server, in `doclens.env`,
`ats.env` and `db.env`.
