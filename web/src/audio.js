// GENESIS sound: procedural Web Audio. Everything here reacts to simulation state; nothing chooses behavior.
// Vela's voice: a tiny breathy two-formant syrinx: a soft glottal tone + breath noise through formants,
// with pitch glides and a faint trill. Silence is the default; vocalizations are rare and state-driven.

const rnd = (a, b) => a + Math.random() * (b - a);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

export class Soundscape {
  constructor() { this.ctx = null; this.enabled = false; this.cool = {}; this.volume = 0.7; }

  async start() {
    if (this.ctx) { await this.ctx.resume(); return; }
    const ctx = this.ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: "interactive" });
    this.master = ctx.createGain(); this.master.gain.value = this.volume;
    const comp = ctx.createDynamicsCompressor(); comp.threshold.value = -18; comp.ratio.value = 3;
    this.master.connect(comp).connect(ctx.destination);
    // shared space: a generated impulse response (no assets)
    this.verb = ctx.createConvolver(); this.verb.buffer = this.impulse(3.2, 2.4);
    this.verbGain = ctx.createGain(); this.verbGain.gain.value = 0.35;
    this.verb.connect(this.verbGain).connect(this.master);
    // world bus: muffled when Vela is inside a thicket (occlusion)
    this.world = ctx.createGain(); this.occl = ctx.createBiquadFilter(); this.occl.type = "lowpass"; this.occl.frequency.value = 16000;
    this.world.connect(this.occl).connect(this.master); this.occl.connect(this.verb);
    this.noise = this.noiseBuffer(4);
    this.buildAmbience(); this.buildPredator();
    this.enabled = true;
  }

  setVolume(v) { this.volume = v; if (this.master) this.master.gain.setTargetAtTime(v, this.ctx.currentTime, 0.05); }
  setMuted(m) { if (this.master) this.master.gain.setTargetAtTime(m ? 0 : this.volume, this.ctx.currentTime, 0.08); }

  impulse(sec, decay) {
    const ctx = this.ctx, n = Math.floor(ctx.sampleRate * sec), b = ctx.createBuffer(2, n, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) { const d = b.getChannelData(ch); for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / n, decay); }
    return b;
  }
  noiseBuffer(sec) {
    const ctx = this.ctx, n = Math.floor(ctx.sampleRate * sec), b = ctx.createBuffer(1, n, ctx.sampleRate), d = b.getChannelData(0);
    let b0 = 0, b1 = 0, b2 = 0;  // pinkish
    for (let i = 0; i < n; i++) { const w = Math.random() * 2 - 1; b0 = 0.997 * b0 + w * 0.029; b1 = 0.985 * b1 + w * 0.032; b2 = 0.95 * b2 + w * 0.048; d[i] = (b0 + b1 + b2 + w * 0.02) * 2.2; }
    return b;
  }
  loopNoise(dest) { const s = this.ctx.createBufferSource(); s.buffer = this.noise; s.loop = true; s.playbackRate.value = rnd(0.9, 1.1); s.connect(dest); s.start(this.ctx.currentTime, rnd(0, 3)); return s; }

  buildAmbience() {
    const ctx = this.ctx;
    // wind through the Fold: pink noise, slow gusts, brighter by day
    this.windF = ctx.createBiquadFilter(); this.windF.type = "lowpass"; this.windF.frequency.value = 500; this.windF.Q.value = 0.4;
    this.windG = ctx.createGain(); this.windG.gain.value = 0.0;
    this.loopNoise(this.windF); this.windF.connect(this.windG).connect(this.world);
    // rain: bright hiss + a lower patter band
    this.rainF = ctx.createBiquadFilter(); this.rainF.type = "highpass"; this.rainF.frequency.value = 1800;
    this.rainG = ctx.createGain(); this.rainG.gain.value = 0;
    this.loopNoise(this.rainF); this.rainF.connect(this.rainG).connect(this.world);
    // the Fold's breath at night: two very slow detuned low tones
    this.droneG = ctx.createGain(); this.droneG.gain.value = 0;
    for (const f of [55, 82.6]) { const o = ctx.createOscillator(); o.type = "sine"; o.frequency.value = f; o.detune.value = rnd(-6, 6); o.connect(this.droneG); o.start(); }
    const df = ctx.createBiquadFilter(); df.type = "lowpass"; df.frequency.value = 220; this.droneG.connect(df).connect(this.world);
    this.nextCritter = 0; this.nextDistant = 0;
  }

  buildPredator() {
    const ctx = this.ctx;
    this.pPan = ctx.createStereoPanner(); this.pG = ctx.createGain(); this.pG.gain.value = 0;
    this.pPan.connect(this.pG).connect(this.world); this.pG.connect(this.verb);
    // low body resonance
    this.pOsc = ctx.createOscillator(); this.pOsc.type = "sawtooth"; this.pOsc.frequency.value = 42;
    this.pLow = ctx.createBiquadFilter(); this.pLow.type = "lowpass"; this.pLow.frequency.value = 110; this.pLow.Q.value = 6;
    this.pOscG = ctx.createGain(); this.pOscG.gain.value = 0.0;
    this.pOsc.connect(this.pLow).connect(this.pOscG).connect(this.pPan); this.pOsc.start();
    // breathing: band-passed noise with a slow amplitude cycle
    this.pBreathF = ctx.createBiquadFilter(); this.pBreathF.type = "bandpass"; this.pBreathF.frequency.value = 380; this.pBreathF.Q.value = 1.4;
    this.pBreathG = ctx.createGain(); this.pBreathG.gain.value = 0;
    this.loopNoise(this.pBreathF); this.pBreathF.connect(this.pBreathG).connect(this.pPan);
    this.pBreathPhase = 0; this.lastPounce = false;
  }

  // --- Vela's voice -------------------------------------------------------
  voice(kind, pan = 0, intensity = 1) {
    if (!this.enabled) return;
    const now = this.ctx.currentTime;
    const spec = VOICE[kind]; if (!spec) return;
    if ((this.cool[kind] || 0) > now) return;
    this.cool[kind] = now + rnd(spec.cool[0], spec.cool[1]);
    const notes = spec.notes();
    let t = now + 0.02;
    for (const n of notes) { this.syllable(t, n, pan, intensity * spec.gain); t += n.d + (n.gap || 0); }
  }

  syllable(t, n, pan, gain) {
    const ctx = this.ctx;
    const out = ctx.createGain(); out.gain.value = 0;
    const p = ctx.createStereoPanner(); p.pan.value = clamp(pan, -0.8, 0.8);
    out.connect(p).connect(this.master); p.connect(this.verb);
    // source: soft glottal tone (triangle + a quiet octave) and breath
    const o = ctx.createOscillator(); o.type = "triangle";
    const o2 = ctx.createOscillator(); o2.type = "sine";
    o.frequency.setValueAtTime(n.f0, t); o.frequency.exponentialRampToValueAtTime(n.f1, t + n.d);
    o2.frequency.setValueAtTime(n.f0 * 2.01, t); o2.frequency.exponentialRampToValueAtTime(n.f1 * 2.01, t + n.d);
    const vib = ctx.createOscillator(); vib.frequency.value = n.trill || rnd(5, 7);
    const vibG = ctx.createGain(); vibG.gain.value = n.trill ? n.f0 * 0.06 : n.f0 * 0.012;
    vib.connect(vibG); vibG.connect(o.frequency); vibG.connect(o2.frequency);
    const o2g = ctx.createGain(); o2g.gain.value = 0.25;
    const src = ctx.createGain();
    o.connect(src); o2.connect(o2g).connect(src);
    const br = ctx.createBufferSource(); br.buffer = this.noise; br.playbackRate.value = 1.6;
    const brF = ctx.createBiquadFilter(); brF.type = "bandpass"; brF.frequency.value = n.f1 * 2.2; brF.Q.value = 2.5;
    const brG = ctx.createGain(); brG.gain.value = n.breath ?? 0.35;
    br.connect(brF).connect(brG).connect(src);
    // two formants give the small, vowel-like "throat"
    const f1 = ctx.createBiquadFilter(); f1.type = "bandpass"; f1.frequency.value = n.fm1 || 1500; f1.Q.value = 4;
    const f2 = ctx.createBiquadFilter(); f2.type = "bandpass"; f2.frequency.value = n.fm2 || 3100; f2.Q.value = 6;
    const f2g = ctx.createGain(); f2g.gain.value = 0.6;
    src.connect(f1).connect(out); src.connect(f2).connect(f2g).connect(out);
    const a = n.a || 0.025, peak = 0.9 * gain * (n.v || 1);
    out.gain.setValueAtTime(0, t); out.gain.linearRampToValueAtTime(peak, t + a);
    out.gain.setTargetAtTime(0, t + Math.max(a, n.d * 0.55), n.d * 0.28);
    const end = t + n.d + 0.6;
    for (const s of [o, o2, vib]) { s.start(t); s.stop(end); }
    br.start(t, rnd(0, 3)); br.stop(end);
    o.onended = () => { out.disconnect(); p.disconnect(); };
  }

  chime(pan = 0, bright = 1) {  // a seed's tiny glassy release
    if (!this.enabled) return;
    const ctx = this.ctx, t = ctx.currentTime + 0.01, g = ctx.createGain(), p = ctx.createStereoPanner();
    p.pan.value = clamp(pan, -0.8, 0.8); g.connect(p).connect(this.world); p.connect(this.verb);
    for (const [f, v] of [[1318.5 * rnd(0.99, 1.01), 0.05], [1975.5, 0.025], [2637, 0.012 * bright]]) {
      const o = ctx.createOscillator(); o.type = "sine"; o.frequency.value = f;
      const e = ctx.createGain(); e.gain.setValueAtTime(0, t); e.gain.linearRampToValueAtTime(v, t + 0.005); e.gain.setTargetAtTime(0, t + 0.01, 0.35);
      o.connect(e).connect(g); o.start(t); o.stop(t + 2);
    }
  }
  thud(pan = 0, freq = 70, gain = 0.25) {
    if (!this.enabled) return;
    const ctx = this.ctx, t = ctx.currentTime + 0.01, o = ctx.createOscillator(), e = ctx.createGain(), p = ctx.createStereoPanner();
    p.pan.value = clamp(pan, -0.8, 0.8);
    o.frequency.setValueAtTime(freq * 1.6, t); o.frequency.exponentialRampToValueAtTime(freq, t + 0.2);
    e.gain.setValueAtTime(gain, t); e.gain.setTargetAtTime(0, t + 0.02, 0.18);
    o.connect(e).connect(p).connect(this.world); p.connect(this.verb); o.start(t); o.stop(t + 1.5);
  }
  splash(pan = 0) {
    if (!this.enabled) return;
    const ctx = this.ctx, t = ctx.currentTime, s = ctx.createBufferSource(), f = ctx.createBiquadFilter(), e = ctx.createGain(), p = ctx.createStereoPanner();
    s.buffer = this.noise; f.type = "bandpass"; f.frequency.setValueAtTime(900, t); f.frequency.exponentialRampToValueAtTime(300, t + 0.4); f.Q.value = 1.2;
    p.pan.value = clamp(pan, -0.8, 0.8); e.gain.setValueAtTime(0.18, t); e.gain.setTargetAtTime(0, t + 0.05, 0.15);
    s.connect(f).connect(e).connect(p).connect(this.world); s.start(t, rnd(0, 3)); s.stop(t + 1);
  }
  critter(dark) {  // sparse glass-like calls of tiny unseen life, mostly at night
    const ctx = this.ctx, t = ctx.currentTime, n = 2 + Math.floor(Math.random() * 3), f = rnd(3800, 6200), p = ctx.createStereoPanner();
    p.pan.value = rnd(-0.9, 0.9); p.connect(this.verb); const g = ctx.createGain(); g.gain.value = 0.012 + 0.02 * dark; g.connect(p);
    for (let i = 0; i < n; i++) {
      const o = ctx.createOscillator(); o.frequency.value = f * rnd(0.98, 1.02); const e = ctx.createGain(); const ti = t + i * rnd(0.07, 0.11);
      e.gain.setValueAtTime(0, ti); e.gain.linearRampToValueAtTime(1, ti + 0.004); e.gain.setTargetAtTime(0, ti + 0.01, 0.02);
      o.connect(e).connect(g); o.start(ti); o.stop(ti + 0.2);
    }
  }
  distantCall(pan) {  // the predator, far beyond sight: a low falling two-part call
    const ctx = this.ctx, t = ctx.currentTime, o = ctx.createOscillator(), f = ctx.createBiquadFilter(), e = ctx.createGain(), p = ctx.createStereoPanner();
    o.type = "sawtooth"; o.frequency.setValueAtTime(96, t); o.frequency.exponentialRampToValueAtTime(58, t + 1.8);
    f.type = "lowpass"; f.frequency.value = 260; f.Q.value = 8; p.pan.value = pan;
    e.gain.setValueAtTime(0, t); e.gain.linearRampToValueAtTime(0.05, t + 0.6); e.gain.setTargetAtTime(0, t + 1.4, 0.5);
    o.connect(f).connect(e).connect(p).connect(this.verb); o.start(t); o.stop(t + 4);
  }

  // --- per-frame update from simulation state ------------------------------
  update(s, dt) {
    if (!this.enabled) return;
    const ctx = this.ctx, now = ctx.currentTime, k = 0.25;
    const dark = 1 - s.light;
    this.windG.gain.setTargetAtTime(0.05 + 0.05 * (0.5 + 0.5 * Math.sin(now * 0.13)) + 0.02 * s.light, now, 1.5);
    this.windF.frequency.setTargetAtTime(300 + 700 * s.light + 200 * Math.sin(now * 0.07), now, 2);
    this.rainG.gain.setTargetAtTime(s.raining ? 0.09 : 0, now, 2.5);
    this.droneG.gain.setTargetAtTime(0.03 * dark * dark, now, 3);
    this.occl.frequency.setTargetAtTime(s.inShelter ? 1100 : 16000, now, 0.6);  // the thicket closes around Vela
    if (now > this.nextCritter) { if (Math.random() < 0.25 + 0.6 * dark && !s.raining) this.critter(dark); this.nextCritter = now + rnd(2.5, 9) / (0.4 + dark); }
    // predator presence
    const p = s.pred;
    if (p.on) {
      const dx = p.x - s.x, dy = p.y - s.y, d = Math.sqrt(dx * dx + dy * dy);
      const pan = clamp(dx / 8, -0.85, 0.85);
      this.pPan.pan.setTargetAtTime(pan, now, 0.2);
      const near = clamp(1 - d / 12, 0, 1);
      const mode = p.mode;
      const loud = mode === "rest" ? 0.15 : mode === "hunt" ? (p.pounce ? 1.0 : 0.25) : mode === "search" ? 0.6 : 0.45;
      this.pG.gain.setTargetAtTime(near * near * loud * 0.6, now, p.pounce ? 0.05 : 0.6);
      this.pOscG.gain.setTargetAtTime(p.pounce ? 0.5 : 0.18, now, 0.1);
      this.pOsc.frequency.setTargetAtTime(p.pounce ? 58 : 40, now, 0.15);
      this.pLow.frequency.setTargetAtTime(p.pounce ? 420 : 120 + 60 * near, now, 0.2);
      this.pBreathPhase += dt * (p.pounce ? 2.2 : mode === "hunt" ? 0.35 : 0.22) * Math.PI * 2;
      this.pBreathG.gain.setTargetAtTime((p.pounce ? 0.7 : 0.35) * Math.max(0, Math.sin(this.pBreathPhase)), now, 0.12);
      if (p.pounce && !this.lastPounce) this.thud(pan, 48, 0.3);
      this.lastPounce = p.pounce;
      if (d > 9 && now > this.nextDistant && dark > 0.4 && mode !== "rest") { this.distantCall(pan); this.nextDistant = now + rnd(25, 60); }
    } else this.pG.gain.setTargetAtTime(0, now, 0.5);
  }
}

// Vela's vocabulary. Pitches are high but soft; formants sit like a tiny vowel "ü".
const VOICE = {
  curious: { cool: [7, 16], gain: 0.22, notes: () => [{ f0: rnd(640, 720), f1: rnd(980, 1150), d: rnd(0.16, 0.22), trill: rnd(24, 30), fm1: 1700, fm2: 3300, gap: 0.05 },
    ...(Math.random() < 0.4 ? [{ f0: 1000, f1: 1250, d: 0.07, v: 0.6 }] : [])] },
  content: { cool: [1.5, 3], gain: 0.2, notes: () => [{ f0: rnd(900, 980), f1: 1100, d: 0.07, a: 0.01, gap: 0.03 }, { f0: 1050, f1: rnd(760, 820), d: 0.16, breath: 0.5, fm1: 1300 }] },
  startle: { cool: [3, 6], gain: 0.3, notes: () => [{ f0: rnd(1500, 1650), f1: rnd(850, 950), d: 0.09, a: 0.004, breath: 0.6, fm1: 2100, fm2: 3600 }] },
  flee: { cool: [2.5, 5], gain: 0.22, notes: () => [{ f0: 1300, f1: 1150, d: 0.05, a: 0.004, gap: 0.04 }, { f0: 1350, f1: 1100, d: 0.05, a: 0.004 }] },
  murmur: { cool: [6, 13], gain: 0.09, notes: () => [{ f0: rnd(430, 480), f1: rnd(380, 420), d: rnd(0.45, 0.7), a: 0.15, breath: 0.85, fm1: 900, fm2: 2400 }] },
  emerge: { cool: [10, 20], gain: 0.16, notes: () => [{ f0: 720, f1: 800, d: 0.08, a: 0.02, gap: rnd(0.35, 0.6), v: 0.6 }, { f0: 760, f1: rnd(960, 1040), d: 0.13, trill: 22 }] },
  sigh: { cool: [12, 25], gain: 0.1, notes: () => [{ f0: 620, f1: 410, d: 0.55, a: 0.08, breath: 0.9, fm1: 1100 }] },
  hungry: { cool: [14, 28], gain: 0.13, notes: () => [{ f0: 560, f1: 700, d: 0.2, gap: 0.12 }, { f0: 600, f1: 520, d: 0.22, breath: 0.5 }] },
};
