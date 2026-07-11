#!/usr/bin/env bash
# Install the nacre memory provider into Hermes — SYMLINK ONLY, NEVER ACTIVATES.
#
# What this does:
#   1. checks the nacre-node addon is built (the sidecar loads it)
#   2. symlinks integrations/hermes/nacre -> $HERMES_HOME/plugins/nacre
#   3. verifies Hermes's own plugin loader discovers it (read-only check)
#
# What this deliberately does NOT do: touch Hermes config. The provider
# stays INACTIVE until you run `hermes memory setup` and select it.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
plugin_src="$here/nacre"
repo_root="$(cd "$here/../.." && pwd)"
hermes_home="${HERMES_HOME:-$HOME/.hermes}"
plugins_dir="$hermes_home/plugins"
link="$plugins_dir/nacre"

# 1. The sidecar needs the built addon.
if [[ ! -f "$repo_root/crates/nacre-node/index.js" ]]; then
  echo "✗ nacre-node addon not built. Build it first:"
  echo "    cd $repo_root/crates/nacre-node && npm install && npm run build"
  exit 1
fi

# 2. Symlink (idempotent; refuses to clobber anything that isn't ours).
mkdir -p "$plugins_dir"
if [[ -L "$link" ]]; then
  current="$(readlink "$link")"
  if [[ "$current" == "$plugin_src" ]]; then
    echo "✓ already installed: $link -> $plugin_src"
  else
    echo "✗ $link is a symlink to something else: $current"
    echo "  remove it yourself if you want it replaced."
    exit 1
  fi
elif [[ -e "$link" ]]; then
  echo "✗ $link exists and is not a symlink — refusing to touch it."
  exit 1
else
  ln -s "$plugin_src" "$link"
  echo "✓ installed: $link -> $plugin_src"
fi

# 3. Read-only discovery check through Hermes's own loader.
hermes_agent="$hermes_home/hermes-agent"
hermes_python=""
for candidate in "$hermes_agent/venv/bin/python" "$hermes_agent/.venv/bin/python"; do
  [[ -x "$candidate" ]] && hermes_python="$candidate" && break
done
if [[ -n "$hermes_python" ]]; then
  if "$hermes_python" - <<PY
import sys
sys.path.insert(0, "$hermes_agent")
from plugins.memory import discover_memory_providers
names = {name for name, _, _ in discover_memory_providers()}
sys.exit(0 if "nacre" in names else 1)
PY
  then
    echo "✓ Hermes's plugin loader discovers 'nacre'"
  else
    echo "✗ Hermes's loader did NOT discover 'nacre' — check $link"
    exit 1
  fi
else
  echo "· skipped loader check (no Hermes venv python found)"
fi

echo
echo "The provider is installed but INACTIVE. Nothing changes until you run:"
echo "    hermes memory setup      # select 'nacre', enter keys"
echo "Kill switches, once active:"
echo "    hermes memory off        # detach the provider"
echo "    rm $hermes_home/nacre/memory.db   # erase the captured graph"
