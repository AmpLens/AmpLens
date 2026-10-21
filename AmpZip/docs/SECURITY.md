# Security

If you expose the AmpZip web service to the public internet, enable authentication and HTTPS before allowing access.

## Authentication

Set `WEBSVC_AUTH_MODE` to `cloudflare` or `basic`, and set `WEBSVC_DEV_ALLOW_NO_AUTH=0`.

For Cloudflare Access, configure the application and permitted email addresses as described in the [deployment guide](DEPLOY.md). For HTTP Basic authentication, use a generated password and keep the credentials private.

## Resource limits

Set `WEBSVC_MAX_CONCURRENT`, `WEBSVC_MAX_QUEUE` and `WEBSVC_RATE_PER_MIN` according to the server's capacity. See [Limits](LIMITS.md) for search budgets and timeout behaviour.

`WEBSVC_JOB_TIMEOUT_S` limits how long the service waits for a job, including queue time. A job marked `timeout` may still be computing. The service does not enforce a memory limit for individual jobs.

## Deployment

Keep the service port accessible only through the configured reverse proxy or tunnel. Run the service as a non-root user and restrict write access to the directories it needs.

Run the service with one worker process. See the [web service documentation](../ampzip/webservice/README.md#service-limitations) for storage and restart behaviour.
