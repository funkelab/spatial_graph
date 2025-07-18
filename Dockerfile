FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git curl tar bzip2 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# auto-detect arch and grab the matching micromamba binary
RUN set -eux; \
    arch="$(uname -m)"; \
    case "$arch" in \
      x86_64)    url_arch=linux-64 ;; \
      aarch64|arm64) url_arch=linux-aarch64 ;; \
      ppc64le)   url_arch=linux-ppc64le ;; \
      *)         echo "Unsupported arch: $arch"; exit 1 ;; \
    esac; \
    curl -Ls "https://micro.mamba.pm/api/micromamba/$url_arch/latest" \
      | tar -xvj -C /usr/local/bin bin/micromamba; \
    chmod +x /usr/local/bin/bin/micromamba; \
    mv /usr/local/bin/bin/micromamba /usr/local/bin/micromamba; \
    rmdir /usr/local/bin/bin


WORKDIR /app
COPY . /app

RUN micromamba create -y -n test-env -c conda-forge python=3.12 pip compilers
RUN micromamba run -n test-env pip install -e . --group test

SHELL ["micromamba", "run", "-n", "test-env", "/bin/bash", "-o", "pipefail", "-c"]


# when you docker run, this will invoke pytest from inside test-env
CMD ["micromamba", "run", "-n", "test-env", "pytest", "-q", "-x"]
