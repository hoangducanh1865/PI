#!/usr/bin/env bash
# Open an SSH server on this Colab runtime, reachable from your machine through a Cloudflare quick tunnel (no account).
# Run by the last cell of notebooks/colab.ipynb; the cell must keep running: stopping it closes the tunnel.
# Login: root with the public keys of the GitHub account (https://github.com/<user>.keys), no password.
# Colab allows SSH only with a paid plan and a positive compute-unit balance (Colab FAQ); free runtimes forbid it.
#
# Author: Duc Anh Hoang
set -euo pipefail
GITHUB_USER=${GITHUB_USER:-hoangducanh1865}

apt-get -qq update > /dev/null
apt-get -qq install -y openssh-server tmux > /dev/null
mkdir -p /root/.ssh /run/sshd
chmod 700 /root/.ssh
curl -fsSL "https://github.com/${GITHUB_USER}.keys" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A > /dev/null
# Our own port: Colab's sshd_config says 2222, which something on the runtime already uses.
PORT=2022
pkill -f "sshd -p ${PORT}" || true   # a re-run replaces the server of the previous run
rm -f /tmp/sshd.log
/usr/sbin/sshd -p "${PORT}" -E /tmp/sshd.log -o ListenAddress=127.0.0.1 -o PasswordAuthentication=no \
  -o PermitRootLogin=prohibit-password
sleep 1
if ! (exec 3<> "/dev/tcp/127.0.0.1/${PORT}") 2> /dev/null; then
  echo "sshd is not listening on 127.0.0.1:${PORT}. Its log:" && cat /tmp/sshd.log && exit 1
fi

# An SSH login starts with a clean environment: give it the notebook's PATH, CUDA and Colab variables.
export -p | grep -E '^declare -x (PATH|LD_LIBRARY_PATH|LIBRARY_PATH|PYTHONPATH|CUDA[A-Z_]*|NV[A-Z_]*|COLAB[A-Z_]*)=' \
  > /root/.colab_env
REPO=$(cd "$(dirname "$0")/.." && pwd)   # an SSH login starts in this repo, wherever it is cloned
grep -q colab_env /root/.bashrc || echo "source /root/.colab_env; cd '${REPO}'" >> /root/.bashrc

curl -fsSL -o /usr/local/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /usr/local/bin/cloudflared
echo "Starting the tunnel (full log: /tmp/cloudflared.log)..."
cloudflared tunnel --no-autoupdate --url "ssh://127.0.0.1:${PORT}" 2>&1 | while read -r line; do
  echo "$line" >> /tmp/cloudflared.log
  if [[ -z ${host:-} && $line =~ https://([a-z0-9-]+\.trycloudflare\.com) ]]; then
    host=${BASH_REMATCH[1]}
    echo "SSH ready. On your machine:  ssh ${host}"
  fi
done
