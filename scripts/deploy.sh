#!/usr/bin/env bash
# Deploys the commit you name to the VM, or HEAD when you name nothing.
#
#   ./scripts/deploy.sh          # deploy the current HEAD
#   ./scripts/deploy.sh <ref>    # deploy that commit or tag
#
# The repository is private and the organisation disables deploy keys, so the
# VM holds no GitHub credential. This ships the committed tree with
# `git archive` over the SSH connection instead, which sends exactly what is
# committed and nothing from the working directory.
#
# `.env` on the VM is never touched: it holds the session secret and the RunPod
# key, and neither is in the repository.
#
# The PDF book is the one thing built here rather than on the VM, because
# printing it needs Chrome and the VM has neither the disk nor the memory to
# spare. The build skips itself when no lesson or layout file has changed.
# To deploy without touching the PDF already on the VM, set SPEECH_SKIP_BOOK=1.

set -euo pipefail

HOST="${SPEECH_HOST:-ifkash@vm.ifkash.dev}"
KEY="${SPEECH_SSH_KEY:-$HOME/.ssh/id_ed25519_kh}"
REMOTE_DIR="${SPEECH_REMOTE_DIR:-docs/speech}"
REF="${1:-HEAD}"
SHA="$(git rev-parse "$REF")"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Warning: the working directory has changes that will not be deployed." >&2
fi

ROOT="$(git rev-parse --show-toplevel)"
if [ "${SPEECH_SKIP_BOOK:-0}" != "1" ]; then
  echo "Building the PDF book"
  node "$ROOT/frontend/scripts/build-book.mjs"
fi

echo "Shipping ${SHA:0:12} to $HOST:~/$REMOTE_DIR"

ssh -i "$KEY" "$HOST" "mkdir -p ~/$REMOTE_DIR"

# --delete would take `.env` and the data volume's bind points with it, so the
# tree is replaced file by file and only tracked paths are overwritten.
git archive --format=tar "$SHA" \
  | ssh -i "$KEY" "$HOST" "tar -x -C ~/$REMOTE_DIR"

# The directory must exist before compose mounts it, or Docker creates it as
# root and the next copy fails.
ssh -i "$KEY" "$HOST" "mkdir -p ~/$REMOTE_DIR/book"
if [ "${SPEECH_SKIP_BOOK:-0}" != "1" ]; then
  echo "Copying the PDF book"
  rsync -az --delete --exclude .build -e "ssh -i $KEY" "$ROOT/book/" "$HOST:$REMOTE_DIR/book/"
fi

echo "Building"
ssh -i "$KEY" "$HOST" "cd ~/$REMOTE_DIR && GIT_SHA=$SHA docker compose up -d --build" 2>&1 \
  | tail -20

# The VM's disk runs close to full, and each rebuild leaves a couple of
# gigabytes of build cache behind. Keep the recent layers and drop the rest.
ssh -i "$KEY" "$HOST" "docker builder prune -f --filter until=72h >/dev/null 2>&1 || true"

echo "Waiting for the health check to report $SHA"
for _ in $(seq 1 60); do
  reported="$(ssh -i "$KEY" "$HOST" \
    "curl -fsS http://127.0.0.1:8096/api/health 2>/dev/null" \
    | python3 -c "import json,sys; print(json.load(sys.stdin).get('commit',''))" 2>/dev/null || true)"
  if [ "$reported" = "$SHA" ]; then
    pages="$(ssh -i "$KEY" "$HOST" \
      "curl -fsS http://127.0.0.1:8096/api/health 2>/dev/null" \
      | python3 -c "import json,sys; print(json.load(sys.stdin).get('book_pages', 0))" 2>/dev/null || echo 0)"
    echo "Live at ${SHA:0:12}, with a PDF book of $pages pages"
    exit 0
  fi
  sleep 5
done

echo "The app did not report $SHA within five minutes. Check the container logs:" >&2
echo "  ssh -i $KEY $HOST 'cd ~/$REMOTE_DIR && docker compose logs --tail 60'" >&2
exit 1
