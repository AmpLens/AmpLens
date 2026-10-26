# Deploying the web service

This guide covers local use, systemd setup and public access. The local examples disable authentication and are intended for use on your own machine.

## 1. Local use

```bash
pip install 'ampzip[web]'

WEBSVC_AUTH_MODE=dev \
WEBSVC_DEV_ALLOW_NO_AUTH=1 \
ampzip-web
```

Alternatively, run Docker Compose from the `AmpZip` directory in a local checkout:

```bash
docker compose up
```

With the default settings, open http://127.0.0.1:8000 on the same machine. To check the service from another terminal:

```bash
curl -s localhost:8000/healthz
```

## 2. Running with systemd

Use the provided systemd unit to start AmpZip at boot and restart it after a failure. Before installing [webservice.service](../ampzip/webservice/deploy/webservice.service), update `User`, `WorkingDirectory`, `EnvironmentFile` and `ExecStart` to match your installation.

The example assumes the service user and `/opt/ampzip` already exist, and `ampzip[web]` is installed in `/opt/ampzip/.venv`. Run the following commands from the `AmpZip` directory, adjusting paths to match your installation:

```bash
sudo install -m 600 ampzip/webservice/deploy/webservice.env.example /opt/ampzip/webservice.env
sudoedit /opt/ampzip/webservice.env
```

Add the settings you need to `webservice.env` and restart the service after editing it. In `basic` mode, this file also contains password hashes. Restrict read access to the service user and administrator.

After configuring the environment file and service unit, install and start the service:

```bash
sudo cp ampzip/webservice/deploy/webservice.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now webservice

systemctl status webservice --no-pager
journalctl -u webservice -f
```

## 3. Public access

For public access, enable `basic` or `cloudflare` authentication and serve the application over HTTPS.

With systemd, add the settings below to `webservice.env`. With Docker, add them to the service's `environment` section in `docker-compose.yml`. The supplied Compose file does not load `webservice.env`. Set `WEBSVC_DEV_ALLOW_NO_AUTH` to `0` explicitly to override the image's development setting.

**HTTP Basic authentication (`basic`)**

Generate a password and its hash with:

```bash
python -m ampzip.webservice.hashpw alice@example.org --gen
```

The command prints a password for the user and an `email:hash` entry for `WEBSVC_BASIC_CREDS`. Replace the example credential below with the generated entry:

```ini
WEBSVC_AUTH_MODE=basic
WEBSVC_DEV_ALLOW_NO_AUTH=0
WEBSVC_BASIC_CREDS=alice@example.org:pbkdf2_sha256$...
```

In Compose YAML, write each `$` in the password hash as `$$`.

Repeated failed login attempts are limited by `WEBSVC_BASIC_MAX_FAILS` and `WEBSVC_BASIC_FAIL_WINDOW_S`. Use a reverse proxy or Cloudflare Tunnel to provide HTTPS.

**Cloudflare Access (`cloudflare`)**

Cloudflare Tunnel connects your server to Cloudflare without requiring a public IP address. Cloudflare Access authenticates users, and AmpZip verifies the Access token on protected API requests.

`ampzip[web]` installs `cryptography` for token verification on all platforms except Windows ARM and Intel macOS. On these platforms, run `pip install cryptography` to build it from source.

1. Install `cloudflared` and log in with `cloudflared tunnel login`.
2. Run `cloudflared tunnel create ampzip`. Record the tunnel ID and the path to its credentials file.
3. Copy [cloudflared-config.example.yml](../ampzip/webservice/deploy/cloudflared-config.example.yml) to `~/.cloudflared/config.yml`. Set the tunnel ID, credentials-file path and hostname. The example routes `simp.YOURDOMAIN.com` to `http://127.0.0.1:8000`.
4. Run `cloudflared tunnel route dns ampzip simp.YOURDOMAIN.com`.
5. In Cloudflare Zero Trust, add an Access application for that hostname with an Allow policy for the intended email addresses. Record the application's AUD tag and your team domain.
6. Add these settings to `webservice.env`:

   ```ini
   WEBSVC_AUTH_MODE=cloudflare
   WEBSVC_DEV_ALLOW_NO_AUTH=0
   CF_ACCESS_TEAM_DOMAIN=YOURTEAM.cloudflareaccess.com
   CF_ACCESS_AUD=<AUD_TAG>
   WEBSVC_EMAIL_WHITELIST=alice@example.org,bob@example.edu
   ```

7. Restart the web service with `sudo systemctl restart webservice`. For Docker, apply these settings to the container environment and recreate the service.
8. Run the tunnel with the provided [cloudflared.service](../ampzip/webservice/deploy/cloudflared.service) unit. Update the service user, executable path and configuration path before installing it. The unit disables automatic updates with `--no-autoupdate`.

   ```bash
   sudo cp ampzip/webservice/deploy/cloudflared.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now cloudflared
   ```

   Do not also run `cloudflared service install` when using this unit.

Open the public URL without an Access session and confirm that it redirects to the login page. After signing in, submit a job and check that its result is returned.

## Deployment checklist

- `WEBSVC_AUTH_MODE` is `basic` or `cloudflare`, and `WEBSVC_DEV_ALLOW_NO_AUTH` is `0`.
- The service port is accessible only through the proxy or tunnel.
- In `cloudflare` mode, `WEBSVC_EMAIL_WHITELIST` matches the Access policy.
- Concurrency, queue and time limits are configured for the server (`WEBSVC_MAX_CONCURRENT`, `WEBSVC_MAX_QUEUE`, `WEBSVC_JOB_TIMEOUT_S`, `WEBSVC_MAX_GBFS_MAX_SECONDS`). See [SECURITY.md](SECURITY.md).
- The service can write `webservice.jsonl` to the configured log directory.

## SFT and CDS

Install `ampzip[sft]` or `ampzip[cds]` in the service's virtual environment. As the service user, run `ampzip fetch-backends`, adding `--cds` for CDS. AmpZip finds the downloaded directory automatically. Set `AMPZIP_BACKENDS` if you use a custom location.

With Docker, run `ampzip fetch-backends` on the host as the user who will run Docker Compose, adding `--cds` for CDS. From the `AmpZip` directory, run:

```bash
docker compose build ampzip          # the ML image builds on ampzip:local
docker compose stop ampzip           # both services publish 127.0.0.1:8000
docker compose up --build ampzip-ml   # mounts the downloaded directory read-only
```

For Docker, set `AMPZIP_BACKENDS_HOST` to the host directory containing `RL_amplitudes_claude` if it differs from the default.
