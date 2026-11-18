// Policy bundle (genesis.policy/1) evaluation: normalization → tanh MLP → logits → argmax. Float64.
export function loadPolicy(bundle, specVersion) {
  if (bundle.schema !== "genesis.policy/1") throw new Error("unknown policy schema");
  if (specVersion && bundle.world_spec_version !== specVersion) throw new Error(`policy spec ${bundle.world_spec_version} != world ${specVersion}`);
  const { mean, var: vr, clip, epsilon } = bundle.normalization;
  const layers = bundle.network.layers;
  const logits = (obs) => {
    let h = new Float64Array(obs.length);
    for (let i = 0; i < obs.length; i++) h[i] = Math.min(clip, Math.max(-clip, (obs[i] - mean[i]) / Math.sqrt(vr[i] + epsilon)));
    for (const L of layers) {
      const out = new Float64Array(L.b.length);
      for (let j = 0; j < L.b.length; j++) {
        let s = 0;
        for (let i = 0; i < h.length; i++) s += h[i] * L.w[i][j];
        out[j] = L.activation === "tanh" ? Math.tanh(s + L.b[j]) : s + L.b[j];
      }
      h = out;
    }
    return h;
  };
  const act = (obs) => { const z = logits(obs); let b = 0; for (let i = 1; i < z.length; i++) if (z[i] > z[b]) b = i; return { action: b, logits: z }; };
  return { logits, act, meta: { milestone: bundle.milestone, step: bundle.training_step, source: bundle.source_checkpoint_id } };
}
