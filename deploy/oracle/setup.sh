#!/usr/bin/env bash
# One-time setup on an Ubuntu 22.04/24.04 Oracle Cloud VM (Ampere A1 recommended: 2 OCPU, 12 GB).
#   curl -fsSLO https://raw.githubusercontent.com/aiori-hackathon/A3-PS003-TC188/main/deploy/oracle/setup.sh && bash setup.sh
set -euo pipefail
sudo apt-get update -y
sudo apt-get install -y docker.io git curl
sudo apt-get install -y docker-compose-v2 || sudo apt-get install -y docker-compose-plugin
sudo systemctl enable --now docker
# Oracle's Ubuntu images block inbound traffic in iptables; open HTTP and HTTPS (the VCN security list must allow them too).
for p in 80 443; do
  sudo iptables -C INPUT -p tcp --dport "$p" -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport "$p" -j ACCEPT
done
sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save || true

[ -d aiori ] || git clone --depth 1 https://github.com/aiori-hackathon/A3-PS003-TC188.git aiori
cd aiori/deploy/oracle
IP=$(curl -fsS https://ifconfig.me)
DOMAIN="${IP//./-}.sslip.io"
if [ ! -f .env ]; then
  read -rsp "Hugging Face READ token (input hidden): " TOKEN; echo
  umask 077
  printf 'HF_TOKEN=%s\nAIORI_ARTIFACTS_REPO=corsenabled/aiori-traffic-artifacts\n' "$TOKEN" > .env
fi
echo "DOMAIN=$DOMAIN" > compose.env
sudo docker compose --env-file compose.env up -d --build
echo
echo "Starting. In 3-5 minutes check: https://$DOMAIN/api/health"
echo "Logs: cd ~/aiori/deploy/oracle && sudo docker compose --env-file compose.env logs -f api"
