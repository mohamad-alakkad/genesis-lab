import { readFileSync } from "node:fs";
import { World } from "../src/sim/world.js";
import { loadPolicy } from "../src/sim/policy.js";
const b = JSON.parse(readFileSync(new URL("../public/policies/policy-best.json", import.meta.url)));
const p = loadPolicy(b, b.world_spec_version);
const w = new World("standard"); w.reset(4000000123);
const t0 = performance.now(); let ticks = 0, eps = 0, food = 0;
while (ticks < 6000) { const { action } = p.act(w.observe()); w.step(action); ticks++; w.events.length = 0;
  if (w.dead || w.truncated) { eps++; food += w.epFood; w.reset(4000000124 + eps); } }
console.log("ms/tick", ((performance.now() - t0) / ticks).toFixed(3), "episodes", eps, "food", food);
