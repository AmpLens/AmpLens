# AmpZip web service: `docker compose up` builds the Rust engine and serves the UI on
# http://127.0.0.1:8000 with no authentication (dev mode). See docs/DEPLOY.md before exposing it.
#
# Stage 1 builds the two wheels (the engine with maturin, the package with hatchling); stage 2
# is a slim runtime that installs both with the web extra. The learned backends are not part of the image: the `ml`
# profile in docker-compose.yml mounts a tree fetched with `ampzip fetch-backends`.

FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir "maturin>=1.5,<2.0" build
WORKDIR /src
COPY . .
# two distributions: the engine extension (maturin) and the pure-Python package (hatchling)
RUN maturin build --release -m ampzip/engine/Cargo.toml --out /wheels \
 && python -m build --wheel --outdir /wheels .

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 10001 ampzip
COPY --from=build /wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/amplens_engine-*.whl "$(ls /tmp/wheels/ampzip-1*.whl)[web]" \
 && rm -rf /tmp/wheels
USER ampzip
WORKDIR /home/ampzip
# Inside the container the service must bind all interfaces; compose publishes it on 127.0.0.1 only.
ENV WEBSVC_HOST=0.0.0.0 WEBSVC_PORT=8000 WEBSVC_AUTH_MODE=dev WEBSVC_DEV_ALLOW_NO_AUTH=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"
ENTRYPOINT ["python", "-m", "ampzip.webservice"]
