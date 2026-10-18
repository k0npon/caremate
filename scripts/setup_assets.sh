#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
ROBOT_DIR="$MODELS/pal_tiago_dual"
SOURCE="${CAREMATE_MENAGERIE_SOURCE:-}"
mkdir -p "$MODELS"

# Copy the TIAGo++ subfolder only. Never write into your dissertation checkout.
if [[ ! -f "$ROBOT_DIR/tiago_dual.xml" ]]; then
  if [[ -n "$SOURCE" && -f "$SOURCE/pal_tiago_dual/tiago_dual.xml" ]]; then
    cp -a "$SOURCE/pal_tiago_dual" "$MODELS/"
    echo "Copied TIAGo++ from $SOURCE"
  else
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT
    git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/google-deepmind/mujoco_menagerie.git "$tmpdir/menagerie"
    git -C "$tmpdir/menagerie" sparse-checkout set pal_tiago_dual
    cp -a "$tmpdir/menagerie/pal_tiago_dual" "$MODELS/"
  fi
fi

if [[ ! -f "$MODELS/humanoid.xml" ]]; then
  curl --fail --location --silent --show-error \
    https://raw.githubusercontent.com/google-deepmind/mujoco/main/model/humanoid/humanoid.xml \
    --output "$MODELS/humanoid.xml"
fi

echo "TIAGo++ ready: $ROBOT_DIR/tiago_dual.xml"
echo "Humanoid ready: $MODELS/humanoid.xml"
echo "Original dissertation files untouched."
