#!/usr/bin/env bash
set -euo pipefail

validate_domain() {
  if [[ ${#preview_domain} -gt 253 || ! "$preview_domain" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*[A-Za-z0-9]$ || "$preview_domain" != *.* ]]; then
    echo "Invalid preview hostname." >&2
    exit 1
  fi
}

if [[ -e .env ]]; then
  preview_domain="$(sed -n 's/^PREVIEW_DOMAIN=//p' .env)"
  validate_domain
  echo "Using existing .env without changing its secrets."
else
  read -r -p "Preview hostname (for example preview.example.com): " preview_domain
  validate_domain
  read -r -s -p "Anthropic API key: " anthropic_key
  printf '\n'
  read -r -s -p "OpenAI API key: " openai_key
  printf '\n'
  if [[ -z "$anthropic_key" || -z "$openai_key" ]]; then
    echo "Both provider API keys are required." >&2
    exit 1
  fi

  api_token="$(openssl rand -hex 32)"
  postgres_password="$(openssl rand -hex 32)"
  minio_password="$(openssl rand -hex 32)"

  umask 077
  cat > .env <<EOF
ANTHROPIC_API_KEY=$anthropic_key
OPENAI_API_KEY=$openai_key
PAPERINTEL_API_AUTH_TOKEN=$api_token
PAPERINTEL_CORS_ALLOW_ORIGINS=https://$preview_domain
PREVIEW_DOMAIN=$preview_domain
POSTGRES_PASSWORD=$postgres_password
MINIO_ROOT_PASSWORD=$minio_password
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=disabled
EOF
  chmod 600 .env
  unset anthropic_key openai_key api_token postgres_password minio_password
fi

sudo docker compose --profile app up -d --build

for attempt in $(seq 1 120); do
  if curl --silent --show-error --fail "https://$preview_domain/health" >/dev/null; then
    echo "PaperIntel is healthy at https://$preview_domain/"
    echo "The preview bearer token is stored in .env; share it with testers over a private channel."
    exit 0
  fi
  sleep 5
done

echo "Preview did not become healthy. Inspect: sudo docker compose --profile app ps" >&2
echo "Logs: sudo docker compose --profile app logs --tail=100" >&2
exit 1
