import { test } from "node:test";
import assert from "node:assert";
import { readFileSync, existsSync } from "node:fs";
import { loadPolicy } from "../src/sim/policy.js";

const fx = new URL("./fixtures/policy.json", import.meta.url);
const bundleUrl = new URL("./fixtures/policy-bundle.json", import.meta.url);
test("policy parity (logits 1e-4, argmax ≥ 99.9%)", { skip: !existsSync(bundleUrl) }, () => {
  const f = JSON.parse(readFileSync(fx, "utf8"));
  const p = loadPolicy(JSON.parse(readFileSync(bundleUrl, "utf8")));
  let same = 0, maxErr = 0;
  f.obs.forEach((o, i) => {
    const { action, logits } = p.act(o);
    logits.forEach((z, j) => { maxErr = Math.max(maxErr, Math.abs(z - f.logits[i][j])); });
    if (action === f.argmax[i]) same++;
  });
  assert.ok(maxErr < 1e-4, `max logit err ${maxErr}`);
  assert.ok(same / f.obs.length >= 0.999, `argmax agreement ${same}/${f.obs.length}`);
});
