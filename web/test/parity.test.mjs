import { test } from "node:test";
import assert from "node:assert";
import { readFileSync, readdirSync } from "node:fs";
import { World } from "../src/sim/world.js";

const dir = new URL("./fixtures/", import.meta.url);
const files = readdirSync(dir).filter((f) => f.startsWith("world-"));

test("constants file identical to Python", () => {
  const a = readFileSync(new URL("../../genesis/world_constants.json", import.meta.url), "utf8");
  const b = readFileSync(new URL("../src/sim/world_constants.json", import.meta.url), "utf8");
  assert.strictEqual(a, b);
});

for (const f of files) {
  test(`world parity ${f}`, () => {
    const fx = JSON.parse(readFileSync(new URL(f, dir), "utf8"));
    const w = new World(fx.family); w.reset(fx.seed);
    let maxErr = 0;
    const cmp = (o, ref, t) => { for (let i = 0; i < ref.length; i++) { const e = Math.abs(o[i] - ref[i]); if (e > maxErr) maxErr = e; assert.ok(e <= 1e-6, `${f} t=${t} obs[${i}] ${o[i]} vs ${ref[i]}`); } };
    cmp(w.observe(), fx.obs[0], 0);
    for (let t = 0; t < fx.actions.length; t++) {
      const ev = fx.events[String(t)];
      if (ev) {
        let ok;
        if (ev[0] === "insert_obstacle") ok = w.insertObstacle(ev[1], ev[2]);
        else if (ev[0] === "wake_vent") ok = w.wakeVent(ev[1]);
        else if (ev[0] === "alert_predator") ok = w.alertPredator();
        else if (ev[0] === "plant") ok = w.plantSeed(ev[1]);
        else if (ev[0] === "raise") ok = w.raiseThicket(ev[1]);
        else if (ev[0] === "rain") ok = w.callRain();
        else if (ev[0] === "lure") ok = w.alertPredator(ev[1]);
        else if (ev[0] === "wake") ok = w.wakeGround(ev[1]);
        else ok = w.moveFood(ev[1], ev[2][0], ev[2][1]);
        assert.strictEqual(ok, fx.accepted[String(t)], `${f} event ${t}`);
      }
      w.step(fx.actions[t]);
      cmp(w.observe(), fx.obs[t + 1], t + 1);
    }
  });
}
