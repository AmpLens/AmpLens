# The web service image with the SFT and CDS dependencies. torch alone is about 2 GB.
# `ampzip fetch-backends` downloads the SFT and CDS code and weights, and docker-compose.yml
# mounts them at /backends. Build the base image first and stop the default service, since
# both publish port 8000.
#   docker compose build ampzip && docker compose stop ampzip && docker compose up --build ampzip-ml
FROM ampzip:local
USER root
RUN pip install --no-cache-dir "torch>=2.0" "tqdm>=4.64" "gymnasium>=0.29" "pandas>=2" \
        "diophantine>=0.2" "wolframclient>=1.1" "gdown>=5"
USER ampzip
