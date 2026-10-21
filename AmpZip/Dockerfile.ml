# The web service image plus the learned-backend extras (torch alone is ~2 GB). The upstream
# code and weights are NOT in this image; mount the tree written by `ampzip fetch-backends` at
# /backends (see docker-compose.yml). Build the base image first, and stop the default
# service before starting this one (both publish port 8000):
#   docker compose build ampzip && docker compose stop ampzip && docker compose up --build ampzip-ml
FROM ampzip:local
USER root
RUN pip install --no-cache-dir "torch>=2.0" "tqdm>=4.64" "gymnasium>=0.29" "pandas>=2" \
        "diophantine>=0.2" "wolframclient>=1.1" "gdown>=5"
USER ampzip
