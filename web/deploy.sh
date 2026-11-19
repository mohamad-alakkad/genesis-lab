#!/usr/bin/env bash
# Deploy the GENESIS web experience as a static-assets Cloudflare Worker (workers.dev, noindex, no custom domain).
# usage: bash web/deploy.sh <worker-name> <policies-dir> [research.json]
#   worker names in use: genesis-preview-3b416cc0d8 (public preview), genesis-staging-* (testing)
# Rollback: node <wrangler.js> rollback --name <worker-name>   (or redeploy a previous tree)
set -euo pipefail
NAME="$1"; POL="$2"; RJ="${3:-web/public/research.json}"
ACCOUNT=44c7f2d2ac955933785da82f8821f68e
WRANGLER=${WRANGLER:-C:/Projects/VEXA-LABS/site/node_modules/wrangler/bin/wrangler.js}
cd "$(dirname "$0")/.."
TMP=$(mktemp -d); SITE="$TMP/site"; mkdir -p "$SITE/src/sim" "$SITE/public/policies" "$SITE/public/assets"
cp web/index.html "$SITE/"
cp web/src/app.js web/src/audio.js web/src/style.css web/src/research.js web/src/research.css web/src/vela-controller.mjs "$SITE/src/"
cp web/src/sim/world.js web/src/sim/policy.js web/src/sim/world_constants.json "$SITE/src/sim/"
cp "$POL"/policy-*.json "$SITE/public/policies/"
[ -f "$RJ" ] && cp "$RJ" "$SITE/public/research.json"
cp web/public/assets/Vela.glb "$SITE/public/assets/"
SPEC=$(python -c "import json;print(json.load(open('web/src/sim/world_constants.json'))['world_spec_version'])")
for f in "$SITE"/public/policies/*.json; do
  v=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['world_spec_version'])" "$f")
  [ "$v" = "$SPEC" ] || { echo "refusing: $(basename $f) is world $v but the site runs $SPEC"; exit 1; }
done
printf '/*\n  X-Robots-Tag: noindex, nofollow, noarchive\n  Referrer-Policy: no-referrer\n  X-Content-Type-Options: nosniff\n' > "$SITE/_headers"
printf 'User-agent: *\nDisallow: /\n' > "$SITE/robots.txt"
cat > "$TMP/wrangler.jsonc" <<JSON
{ "name": "$NAME", "account_id": "$ACCOUNT", "compatibility_date": "2026-11-01",
  "workers_dev": true, "preview_urls": false,
  "assets": { "directory": "./site", "not_found_handling": "404-page" } }
JSON
( cd "$TMP" && node "$WRANGLER" deploy --config wrangler.jsonc > deploy.log 2>&1 ) || { tail -20 "$TMP/deploy.log"; exit 1; }
grep -E "Version ID" "$TMP/deploy.log"
echo "https://$NAME.dentexa-com.workers.dev  (world $SPEC)"
rm -rf "$TMP"
