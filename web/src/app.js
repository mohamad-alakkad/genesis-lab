// GENESIS observation layer. The simulation (sim/world.js) and the trained policy (sim/policy.js) are the
// only sources of behavior. Rendering and sound interpolate and express; they never choose actions.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { World, CONSTANTS as C, SPEC_VERSION, RAY_REL, N_CH } from "./sim/world.js";
import { loadPolicy } from "./sim/policy.js";
import { createVelaController } from "./vela-controller.mjs";
import { Soundscape } from "./audio.js";

const W = C.width, H = C.height, TICK = 0.1;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const isPhone = matchMedia("(max-width: 650px)").matches;
const params = new URLSearchParams(location.search);
const lite = params.has("lite") || isPhone;  // quality tier: standard materials instead of physical (phones, software GL)
const $ = (id) => document.getElementById(id);
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const angLerp = (a, b, t) => { let d = ((b - a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI; return a + d * t; };
const toX = (x) => x - W / 2, toZ = (y) => y - H / 2;
const yawOf = (h) => Math.atan2(-Math.cos(h), -Math.sin(h));  // GLB forward is −Z
const pal = { abyss: 0x070d17, ocean: 0x102d39, life: 0x46e7d7, neural: 0x8e73ff, energy: 0xffb36b, hazard: 0xf2978a };

// ------------------------------------------------------------------ renderer / camera
const canvas = $("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: !lite, powerPreference: lite ? "default" : "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, isPhone ? 1.5 : 1.75));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(pal.abyss, 44, 78);  // camera sits ~40 u away: fog only the far ribs
const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 200);
const ELEV = THREE.MathUtils.degToRad(35);
let viewH = isPhone ? 5.0 : 5.6;  // keeps Vela ~110–200 CSS px wide (interface spec)
const camTarget = new THREE.Vector3();
function resize() {
  const w = innerWidth, h = innerHeight, a = w / h;
  renderer.setSize(w, h, false);
  const vh = a < 0.8 ? viewH * 1.25 : viewH;
  cam.left = -vh * a / 2; cam.right = vh * a / 2; cam.top = vh / 2; cam.bottom = -vh / 2; cam.updateProjectionMatrix();
}
addEventListener("resize", resize); resize();
function placeCamera() { cam.position.set(camTarget.x, camTarget.y + Math.sin(ELEV) * 40, camTarget.z + Math.cos(ELEV) * 40); cam.lookAt(camTarget); }

// ------------------------------------------------------------------ light
const hemi = new THREE.HemisphereLight(0xbfe6ee, 0x1a3240, 1.0); scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff1dc, 1.6); sun.position.set(12, 18, -10); scene.add(sun); scene.add(sun.target);
const velaGlow = new THREE.PointLight(pal.life, 0.9, 3.6, 1.4); scene.add(velaGlow);

// ------------------------------------------------------------------ ground (procedural, extends beyond bounds)
const groundU = { uTime: { value: 0 }, uLight: { value: 1 }, uRain: { value: 0 }, uMeadows: { value: [new THREE.Vector2(), new THREE.Vector2(), new THREE.Vector2()] },
  uPools: { value: [new THREE.Vector3(-99, -99, 0), new THREE.Vector3(-99, -99, 0)] }, uBounds: { value: new THREE.Vector2(W / 2, H / 2) } };
const ground = new THREE.Mesh(new THREE.PlaneGeometry(90, 70, 1, 1), new THREE.ShaderMaterial({
  uniforms: groundU, fog: false,
  vertexShader: `varying vec2 vP; void main(){ vec4 w = modelMatrix*vec4(position,1.); vP = w.xz; gl_Position = projectionMatrix*viewMatrix*w; }`,
  fragmentShader: `
    uniform float uTime, uLight, uRain; uniform vec2 uMeadows[3]; uniform vec3 uPools[2]; uniform vec2 uBounds; varying vec2 vP;
    float h(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453); }
    float n(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.-2.*f); return mix(mix(h(i),h(i+vec2(1,0)),f.x),mix(h(i+vec2(0,1)),h(i+vec2(1,1)),f.x),f.y); }
    float fbm(vec2 p){ float s=0., a=.5; for(int i=0;i<5;i++){ s+=a*n(p); p*=2.03; a*=.5; } return s; }
    void main(){
      vec3 abyss=vec3(.027,.051,.09), ocean=vec3(.063,.176,.224), moss=vec3(.10,.22,.20), sand=vec3(.20,.24,.25);
      float f = fbm(vP*.35), g = fbm(vP*1.7+3.1);
      vec3 col = mix(ocean*0.8, sand*0.7, f*0.55) + (g-.5)*0.05;
      float m = 0.; for(int i=0;i<3;i++){ m += smoothstep(3.2, 0.4, length(vP-uMeadows[i])); }
      col = mix(col, moss*1.25, clamp(m,0.,1.)*0.55);
      float wet = 0.;
      for(int i=0;i<2;i++){ float d = length(vP-uPools[i].xy)-uPools[i].z; wet = max(wet, smoothstep(0.8, 0., d)); }
      col = mix(col, col*0.55 + vec3(0.,0.03,0.05), wet*0.6);
      vec2 e = abs(vP) - uBounds; float out_ = max(e.x, e.y);
      col *= mix(1., 0.32, smoothstep(-0.4, 6., out_));                // the Fold continues into darkness
      float day = uLight;
      col *= mix(vec3(.50,.60,.85), vec3(1.15,1.08,1.0), day);          // moonlit blue → warm day
      col *= 1. - 0.12*uRain;
      col += vec3(.02,.06,.07)*uRain*fbm(vP*4.+uTime*.5)*0.6;            // rain sheen
      gl_FragColor = vec4(col, 1.);
      #include <colorspace_fragment>
    }`,
}));
ground.rotation.x = -Math.PI / 2; scene.add(ground);

// distant ribs of the Fold: tall dark forms beyond the bounds, and the bright upper-right arch
const ribs = new THREE.Group(); scene.add(ribs);
{
  const ribMat = new THREE.MeshStandardMaterial({ color: 0x0d1c26, roughness: 0.9, metalness: 0.05 });
  const rng = mulberry(7);
  for (let i = 0; i < 26; i++) {
    const side = [0, 1, 3][i % 3], t = rng() * 1.2 - 0.1, hgt = 3 + rng() * 7;  // never on the camera side
    const g = new THREE.CylinderGeometry(0.15 + rng() * 0.3, 0.6 + rng() * 0.9, hgt, 7, 1);
    const m = new THREE.Mesh(g, ribMat);
    const px = side === 0 ? lerp(-W / 2 - 4, W / 2 + 4, t) : side === 1 ? W / 2 + 6 + rng() * 5 : -W / 2 - 6 - rng() * 5;
    const pz = side === 0 ? -H / 2 - 5 - rng() * 6 : lerp(-H / 2 - 3, H / 2 - 2, t);
    m.position.set(px, hgt / 2 - 0.3, pz); m.rotation.z = (rng() - 0.5) * 0.35; m.rotation.x = (rng() - 0.5) * 0.25;
    ribs.add(m);
  }
  const arch = new THREE.Mesh(new THREE.TorusGeometry(5, 0.35, 8, 40, Math.PI), new THREE.MeshBasicMaterial({ color: 0x9feee4, fog: false, transparent: true, opacity: 0.35 }));
  arch.position.set(W / 2 + 6, 0, -H / 2 - 9); arch.rotation.y = -0.6; ribs.add(arch);
}

function mulberry(seed) { let s = seed >>> 0; return () => { s = (s + 0x6D2B79F5) >>> 0; let t = Math.imul(s ^ (s >>> 15), s | 1); t = ((t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t) >>> 0; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

// ------------------------------------------------------------------ world props (rebuilt per episode)
const props = new THREE.Group(); scene.add(props);
const slateMat = new THREE.MeshStandardMaterial({ color: 0x6b8794, roughness: 0.7, metalness: 0.05, flatShading: true });
const frondMat = new THREE.MeshStandardMaterial({ color: 0x3f9a86, roughness: 0.55, side: THREE.DoubleSide, emissive: 0x124a42, emissiveIntensity: 0.5 });
const waterMat = new THREE.ShaderMaterial({ transparent: true, uniforms: { uTime: groundU.uTime, uLight: groundU.uLight, uRain: groundU.uRain, uRipple: { value: new THREE.Vector4(0, 0, -10, 0) } },
  vertexShader: `varying vec2 vU; varying vec2 vW; void main(){ vU = uv; vec4 w = modelMatrix*vec4(position,1.); vW = w.xz; gl_Position = projectionMatrix*viewMatrix*w; }`,
  fragmentShader: `uniform float uTime,uLight,uRain; uniform vec4 uRipple; varying vec2 vU; varying vec2 vW;
    void main(){ float r = length(vU-.5)*2.; if(r>1.) discard;
      float edge = smoothstep(1., .82, r);
      float rip = sin(r*24.-uTime*1.6)*.5+.5;
      float rr = length(vW-uRipple.xy); float age = uTime-uRipple.z; float ring = exp(-pow((rr-age*0.9)*5.,2.))*exp(-age*1.2);
      float rain = uRain*step(.985, fract(sin(dot(floor(vW*6.+floor(uTime*3.)), vec2(12.9898,78.233)))*43758.5453));
      vec3 deep = mix(vec3(.02,.06,.10), vec3(.05,.2,.24), uLight);
      vec3 col = deep + vec3(.15,.55,.55)*(rip*.04 + ring*.5 + rain*.4) + vec3(.4,.9,.9)*(1.-edge)*.08;
      gl_FragColor = vec4(col, .82*edge + .1);
      #include <colorspace_fragment>
    }` });
const seedMat = new THREE.MeshStandardMaterial({ color: 0xffd3a1, emissive: pal.energy, emissiveIntensity: 1.6, roughness: 0.3 });
const haloTex = makeHalo();
function makeHalo() { const c = document.createElement("canvas"); c.width = c.height = 64; const g = c.getContext("2d"); const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32); gr.addColorStop(0, "rgba(255,255,255,1)"); gr.addColorStop(0.3, "rgba(255,255,255,.35)"); gr.addColorStop(1, "rgba(255,255,255,0)"); g.fillStyle = gr; g.fillRect(0, 0, 64, 64); const t = new THREE.CanvasTexture(c); return t; }
function halo(color, size, opacity) { const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: haloTex, color, transparent: true, opacity, depthWrite: false, blending: THREE.AdditiveBlending })); s.scale.set(size, size, 1); return s; }
const shadowTex = (() => { const c = document.createElement("canvas"); c.width = c.height = 64; const g = c.getContext("2d"); const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32); gr.addColorStop(0, "rgba(0,0,0,.6)"); gr.addColorStop(1, "rgba(0,0,0,0)"); g.fillStyle = gr; g.fillRect(0, 0, 64, 64); return new THREE.CanvasTexture(c); })();
function contactShadow(size) { const m = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshBasicMaterial({ map: shadowTex, transparent: true, depthWrite: false })); m.rotation.x = -Math.PI / 2; m.position.y = 0.01; return m; }

let view = null;  // per-episode visual objects
function buildProps(world) {
  props.clear();
  view = { slates: [], seeds: [], vents: [], fronds: [], pools: [] };
  const rng = mulberry(world.seed ^ 0x9e3779b9);
  world.meadows.forEach((m, i) => groundU.uMeadows.value[i].set(toX(m.x), toZ(m.y)));
  groundU.uPools.value.forEach((v) => v.set(-99, -99, 0));
  world.pools.forEach((p, i) => {
    if (!p.on) return;
    groundU.uPools.value[i].set(toX(p.x), toZ(p.y), p.r);
    const m = new THREE.Mesh(new THREE.CircleGeometry(p.r, 48), waterMat); m.rotation.x = -Math.PI / 2; m.position.set(toX(p.x), 0.015, toZ(p.y)); props.add(m);
  });
  world.slates.forEach((o, j) => { view.slates[j] = null; if (o.on) addSlate(o, j, rng); });
  // thickets: many leaning fronds around each shelter circle; they part and sway (wind + Vela)
  const frondGeo = new THREE.PlaneGeometry(0.16, 1, 1, 4); frondGeo.translate(0, 0.5, 0);
  const shelters = world.shelters.filter((s) => s.on);
  const count = shelters.reduce((a, s) => a + Math.round(70 * s.r * s.r), 0);
  const inst = new THREE.InstancedMesh(frondGeo, frondMat, Math.max(count, 1));
  const base = [];
  let k = 0;
  for (const s of shelters) {
    const n = Math.round(70 * s.r * s.r);
    for (let i = 0; i < n; i++) {
      const a = rng() * Math.PI * 2, r = Math.sqrt(rng()) * s.r * 1.02;
      const x = toX(s.x) + Math.cos(a) * r, z = toZ(s.y) + Math.sin(a) * r;
      const hgt = 0.5 + rng() * 0.75 * (1 - 0.4 * r / s.r), lean = 0.25 + rng() * 0.35, yaw = rng() * Math.PI * 2;
      base.push({ x, z, hgt, lean, yaw, ph: rng() * 6.28 }); k++;
    }
    const sh = contactShadow(s.r * 2.6); sh.position.set(toX(s.x), 0.012, toZ(s.y)); sh.material.opacity = 0.7; props.add(sh);
  }
  inst.count = k; inst.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  view.fronds = { inst, base }; props.add(inst);
  world.vents.forEach((v, i) => {
    if (!v.present) return;
    const g = new THREE.Group(); g.position.set(toX(v.x), 0, toZ(v.y));
    const segs = 9;
    for (let s = 0; s < segs; s++) {
      if (s % 3 === 1) continue;  // the broken coral ring
      const a0 = s / segs * Math.PI * 2;
      const seg = new THREE.Mesh(new THREE.TorusGeometry(v.r, 0.05, 5, 6, Math.PI * 2 / segs * 0.8), new THREE.MeshStandardMaterial({ color: pal.hazard, emissive: pal.hazard, emissiveIntensity: 0.2, roughness: 0.5 }));
      seg.rotation.x = -Math.PI / 2; seg.rotation.z = a0; seg.position.y = 0.04; g.add(seg);
    }
    const glow = new THREE.Mesh(new THREE.CircleGeometry(v.r, 40), new THREE.MeshBasicMaterial({ color: pal.hazard, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending }));
    glow.rotation.x = -Math.PI / 2; glow.position.y = 0.02; g.add(glow);
    props.add(g); view.vents[i] = { g, glow, segs: g.children.filter((c) => c !== glow), on: 0 };
  });
  world.food.forEach((f, i) => {
    const g = new THREE.Group();
    const core = new THREE.Mesh(new THREE.IcosahedronGeometry(0.13, 1), seedMat); core.position.y = 0.16; g.add(core);
    const h = halo(pal.energy, 0.9, 0.55); h.position.y = 0.18; g.add(h);
    g.visible = f.on; g.scale.setScalar(f.on ? 1 : 0.001); props.add(g);
    view.seeds[i] = { g, core, s: f.on ? 1 : 0, ph: rng() * 6.28 };
  });
}
function addSlate(o, j, rng = Math.random) {
  const geo = new THREE.CylinderGeometry(o.r * 0.92, o.r, 0.22 + o.r * 0.25, 9, 1);
  const m = new THREE.Mesh(geo, slateMat); m.position.set(toX(o.x), 0.12, toZ(o.y)); m.rotation.y = rng() * 6.28; m.scale.y = 0.9 + rng() * 0.4;
  const top = new THREE.Mesh(new THREE.CylinderGeometry(o.r * 0.6, o.r * 0.85, 0.12, 7), slateMat); top.position.y = 0.17; top.rotation.y = 0.4; m.add(top);
  const sh = contactShadow(o.r * 2.8); sh.position.set(toX(o.x), 0.011, toZ(o.y)); props.add(sh);
  props.add(m); view.slates[j] = m;
}

// ------------------------------------------------------------------ the predator ("Shade"): a long low body of linked segments
const shade = new THREE.Group(); scene.add(shade);
const shadeMat = lite ? new THREE.MeshStandardMaterial({ color: 0x0b1015, roughness: 0.4, metalness: 0.3 })
  : new THREE.MeshPhysicalMaterial({ color: 0x0b1015, roughness: 0.35, metalness: 0.2, clearcoat: 0.6, iridescence: 0.5, sheen: 0.4, sheenColor: new THREE.Color(0x3a2a55) });
const SEG = 9, shadeSegs = [], shadeTrail = [];
for (let i = 0; i < SEG; i++) {
  const s = new THREE.Mesh(new THREE.SphereGeometry(1, 14, 10), shadeMat);
  const t = i / (SEG - 1), r = 0.3 * (1 - 0.55 * t) * (i === 0 ? 0.95 : 1);
  s.scale.set(r * 1.25, r * 0.62, r * 1.5); shade.add(s); shadeSegs.push({ m: s, r });
}
const eyeMat = new THREE.MeshBasicMaterial({ color: 0xe8f2c8, transparent: true, opacity: 0.2 });
const eyes = [-1, 1].map((sd) => { const e = new THREE.Mesh(new THREE.SphereGeometry(0.035, 8, 6), eyeMat); e.scale.set(1.6, 0.6, 0.8); e.userData.sd = sd; shade.add(e); return e; });
const shadeShadow = contactShadow(2.2); scene.add(shadeShadow);

// ------------------------------------------------------------------ Vela
let vela = null, velaRoot = new THREE.Group(), velaChild = new THREE.Group(); scene.add(velaRoot); velaRoot.add(velaChild);
const velaShadow = contactShadow(1.0); scene.add(velaShadow);
const velaHalo = halo(pal.life, 1.4, 0.18); velaRoot.add(velaHalo); velaHalo.position.y = 0.2;

// ------------------------------------------------------------------ rain and motes
const RAIN_N = isPhone ? 260 : 600;
const rainGeo = new THREE.BufferGeometry(); const rainPos = new Float32Array(RAIN_N * 6); rainGeo.setAttribute("position", new THREE.BufferAttribute(rainPos, 3));
const rainMat = new THREE.LineBasicMaterial({ color: 0x9fd7e0, transparent: true, opacity: 0 });
const rain = new THREE.LineSegments(rainGeo, rainMat); scene.add(rain); rain.frustumCulled = false;
const rainSeeds = Array.from({ length: RAIN_N }, () => [Math.random() * 30 - 15, Math.random() * 12, Math.random() * 30 - 15]);
const MOTE_N = isPhone ? 32 : 90;
const moteGeo = new THREE.BufferGeometry(); const motePos = new Float32Array(MOTE_N * 3); moteGeo.setAttribute("position", new THREE.BufferAttribute(motePos, 3));
const motes = new THREE.Points(moteGeo, new THREE.PointsMaterial({ size: 0.06, color: 0xbff7ee, transparent: true, opacity: 0.5, depthWrite: false, blending: THREE.AdditiveBlending })); scene.add(motes); motes.frustumCulled = false;
const moteSeeds = Array.from({ length: MOTE_N }, () => [Math.random() * 20 - 10, 0.2 + Math.random() * 2.5, Math.random() * 14 - 7, Math.random() * 6.28]);

// ------------------------------------------------------------------ mind view (what Vela senses)
const mindGeo = new THREE.BufferGeometry(); const mindPos = new Float32Array(C.n_rays * 2 * 3); const mindCol = new Float32Array(C.n_rays * 2 * 3);
mindGeo.setAttribute("position", new THREE.BufferAttribute(mindPos, 3)); mindGeo.setAttribute("color", new THREE.BufferAttribute(mindCol, 3));
const mind = new THREE.LineSegments(mindGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.75 })); mind.visible = false; mind.frustumCulled = false; scene.add(mind);

// ------------------------------------------------------------------ simulation state
const sound = new Soundscape();
let world, policies = {}, brainKey = null, brain = null;
let prev = null, cur = null, acc = 0, speed = 1, paused = false, episode = 0, lastProbs = null, lastAction = 7;
let behave = { label: "", since: 0, prevHeard: false, prevShelter: false, wasLow: false, closing: 0, lastFoodD: Infinity, idleT: 0, threatT: 0 };
let tool = null, endTimer = 0;

async function loadBrains() {
  const names = ["final", "best", "middle", "early", "step0"];
  for (const n of names) {
    try {
      const r = await fetch(`public/policies/policy-${n}.json`); if (!r.ok) continue;
      const b = await r.json(); policies[n] = { bundle: b, p: loadPolicy(b, SPEC_VERSION) };
    } catch (e) { /* missing or incompatible bundle: not offered */ }
  }
  const sel = $("brain");
  const label = { final: "trained (final)", best: "trained (best so far)", middle: "half-trained", early: "early learner", step0: "untrained (random weights)" };
  for (const n of names) if (policies[n]) { const o = document.createElement("option"); o.value = n; o.textContent = label[n]; sel.appendChild(o); }
  brainKey = sel.value; setBrain(brainKey);
  sel.onchange = () => setBrain(sel.value);
}
function setBrain(k) {
  brainKey = k; brain = policies[k]?.p || null;
  const m = brain?.meta;
  $("brainTag").textContent = m ? `${k} · step ${m.step.toLocaleString()}` : "no brain loaded";
  $("honesty").textContent = m ? `Every movement is chosen by this network (${m.source}), one decision per tick. Labels below are derived from the simulation, not feelings.` : "No trained policy available.";
}

function newEpisode() {
  world = new World("standard");
  world.reset(4000000000 + Math.floor(Math.random() * 100000000));
  episode++;
  buildProps(world);
  prev = cur = snap();
  camTarget.set(toX(world.x), 0, toZ(world.y)); placeCamera();
  shadeTrail.length = 0;
  behave = { label: "", since: 0, prevHeard: false, prevShelter: world.inShelter, wasLow: false, closing: 0, lastFoodD: Infinity, idleT: 0, threatT: 0 };
  endTimer = 0;
}
function snap() { const s = world.snapshot(); s.seedOn = world.food.map((f) => f.on); s.seedPos = world.food.map((f) => [f.x, f.y]); return s; }

function decide() {
  const obs = world.observe();
  if (!brain) return { a: 7, obs, probs: null };
  const { action, logits } = brain.act(obs);
  const mx = Math.max(...logits); const ex = Array.from(logits, (z) => Math.exp(z - mx)); const sum = ex.reduce((a, b) => a + b, 0);
  return { a: action, obs, probs: ex.map((e) => e / sum) };
}

function tick() {
  if (world.dead || world.truncated) return;
  const { a, obs, probs } = decide();
  lastProbs = probs; lastAction = a;
  const r = world.step(a);
  prev = cur; cur = snap(); cur.obs = obs; cur.action = a; cur.dashing = r.dashing; cur.probs = probs;
  handleEvents(world.events.splice(0));
  interpretBehavior(obs, probs);
  if (world.dead || world.truncated) endEpisode();
}

function endEpisode() {
  const days = (world.t / C.day_ticks).toFixed(1);
  const msg = world.caught ? `Vela was caught after ${days} days. The Fold goes quiet.` : world.dead ? `Vela ran out of energy after ${days} days.` : `Vela survived the whole span: ${days} days.`;
  toast(msg + " A new life begins.", 4200);
  if (world.caught) sound.thud(0, 40, 0.35);
  endTimer = 4.5;
}

// ------------------------------------------------------------------ events → visuals + sound
let lastRipple = -10;
function panOf(x) { return clamp((toX(x) - camTarget.x) / 6, -1, 1); }
function handleEvents(evs) {
  for (const e of evs) {
    if (e.kind === "eat") { velaEvents.push({ id: e.id, clip: "Collect" }); sound.chime(panOf(e.x)); sound.voice("content", panOf(world.x)); seedFx(e.index, 0); }
    else if (e.kind === "grow") seedFx(e.index, 1);
    else if (e.kind === "enter_shelter") { /* soft brush of fronds */ }
    else if (e.kind === "leave_shelter") { if (behave.threatT > 2) sound.voice("emerge", panOf(world.x)); }
    else if (e.kind === "rain_start") toast("Rain begins. It hides Vela from sight, and blurs what Vela can hear.", 3500);
    else if (e.kind === "predator_mode" && e.to === "hunt") { if (!world.inShelter) sound.voice("startle", panOf(world.x)); }
  }
}
function seedFx(i, grow) { const s = view.seeds[i]; if (!s) return; s.target = grow; if (grow) { s.g.visible = true; s.s = 0.001; } }

// derived, evidence-based labels (never inputs to the policy)
const velaEvents = [];
function interpretBehavior(obs, probs) {
  const S = N_CH * C.n_rays, b = behave, p = world.pred;
  const heard = obs[S + 12] > 0;
  let seesPred = false, seesFood = false, nearestFood = Infinity;
  for (let k = 0; k < C.n_rays; k++) { if (obs[N_CH * k + 4] > 0) seesPred = true; if (obs[N_CH * k + 1] > 0) { seesFood = true; nearestFood = Math.min(nearestFood, 6 * (1 - obs[N_CH * k + 1])); } }
  const threat = heard || seesPred;
  b.threatT = threat ? b.threatT + TICK : Math.max(0, b.threatT - TICK * 0.5);
  b.closing = seesFood && nearestFood < b.lastFoodD - 0.02 ? b.closing + 1 : 0; b.lastFoodD = nearestFood;
  b.idleT = world.v < 0.02 ? b.idleT + TICK : 0;
  const ent = probs ? -probs.reduce((a, q) => a + (q > 0 ? q * Math.log(q) : 0), 0) : 0;
  let label;
  if (world.resting && world.inShelter && !threat) label = "Resting, sheltered";
  else if (world.inShelter && threat) label = "Hiding";
  else if (cur.dashing && threat) label = "Fleeing";
  else if (threat && world.v < 0.07) label = "Frozen, watching";
  else if (threat) label = "Alert";
  else if (world.resting) label = "Resting";
  else if (b.idleT > 0.6 && ent > 1.4) label = "Hesitating";
  else if (b.closing >= 3) label = "Nearing seed";
  else if (world.v > 0.02) label = world.energy < 0.25 ? "Searching, hungry" : seesFood ? "Foraging" : "Exploring";
  else label = "Still";
  if (world.stamina < 0.2) label += " · exhausted";
  if (label !== b.label) { b.label = label; b.since = 0; }
  // voice: rare, tied to real state changes
  const pan = panOf(world.x);
  if (heard && !b.prevHeard && !world.inShelter) sound.voice("startle", pan, 0.8);
  if (label.startsWith("Fleeing")) sound.voice("flee", pan);
  if (label === "Resting, sheltered" || label === "Resting") sound.voice("murmur", pan);
  if (label === "Exploring" && Math.random() < 0.004) sound.voice("curious", pan);
  if (world.energy < 0.2 && Math.random() < 0.003) sound.voice("hungry", pan);
  if (world.stamina < 0.15 && Math.random() < 0.01) sound.voice("sigh", pan);
  if (world.inWater && !prev.inWater) { sound.splash(pan); lastRipple = groundU.uTime.value; waterMat.uniforms.uRipple.value.set(toX(world.x), toZ(world.y), lastRipple, 0); }
  // Vela clip events
  if (b.wasLow && world.energy > 0.3) { velaEvents.push({ id: `rec${world.t}`, clip: "Recover" }); b.wasLow = false; }
  if (world.energy < 0.2) b.wasLow = true;
  if (threat && Math.abs(world.w) > 0.15 && !world.inShelter && Math.random() < 0.15) velaEvents.push({ id: `av${world.t}`, clip: world.w < 0 ? "Avoid_L" : "Avoid_R" });
  b.prevHeard = heard; b.prevShelter = world.inShelter;
  const el = $("stateLabel"); el.textContent = label; el.classList.toggle("danger", threat && !world.inShelter);
}

// ------------------------------------------------------------------ interaction
const ray = new THREE.Raycaster(), plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), hit = new THREE.Vector3();
document.querySelectorAll(".tools button").forEach((b) => b.addEventListener("click", () => {
  tool = tool === b.dataset.tool ? null : b.dataset.tool;
  document.querySelectorAll(".tools button").forEach((x) => x.setAttribute("aria-pressed", x.dataset.tool === tool ? "true" : "false"));
  if (tool) toast({ seed: "Tap open ground to move a seed there.", slate: "Tap open ground to place a slate.", vent: "Tap a vent ring to wake it.", sound: "Tap anywhere: the predator will come to search where Vela is." }[tool], 2400);
}));
addEventListener("keydown", (e) => { if (e.key === "Escape") { tool = null; document.querySelectorAll(".tools button").forEach((x) => x.setAttribute("aria-pressed", "false")); } if (e.key === " ") { paused = !paused; toast(paused ? "Paused" : "Resumed", 900); e.preventDefault(); } });
canvas.addEventListener("pointerdown", (e) => {
  if (!tool || !world) return;
  const r = canvas.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), cam);
  if (!ray.ray.intersectPlane(plane, hit)) return;
  const sx = hit.x + W / 2, sy = hit.z + H / 2;
  let ok = false, why = "";
  if (tool === "seed") {
    let best = -1, bd = -1;
    world.food.forEach((f, i) => { if (f.on) { const d = (f.x - world.x) ** 2 + (f.y - world.y) ** 2; if (d > bd) { bd = d; best = i; } } });
    ok = best >= 0 && world.moveFood(best, sx, sy);
    why = best < 0 ? "No ripe seed to move right now." : "Seeds only grow on open ground (not in slates, thickets, water or vents, and not right under Vela).";
    if (ok) { view.seeds[best].g.position.set(toX(sx), 0, toZ(sy)); sound.chime(panOf(sx), 0.5); }
  } else if (tool === "slate") {
    const before = world.slates.findIndex((o) => !o.on);
    ok = world.insertObstacle(0.6, [sx, sy]);
    why = before < 0 ? "The Fold holds no more slates." : "A slate needs clear ground: away from Vela, the predator, seeds, thickets and water.";
    if (ok) { addSlate(world.slates[before], before); sound.thud(panOf(sx), 90, 0.18); }
  } else if (tool === "vent") {
    let best = -1, bd = 2.25;
    world.vents.forEach((v, i) => { if (v.present) { const d = (v.x - sx) ** 2 + (v.y - sy) ** 2 - v.r * v.r; if (d < bd) { bd = d; best = i; } } });
    ok = best >= 0 && world.wakeVent(best); why = "Tap closer to a vent ring.";
  } else if (tool === "sound") {
    ok = world.alertPredator(); why = world.pred.mode === "rest" || world.pred.mode === 3 ? "The predator is resting and does not stir." : "Nothing hunts here now.";
    sound.thud(panOf(sx), 160, 0.12);
  }
  if (!ok) toast(why, 2600);
  cur = snap(); prev = prev || cur;
});

let toastT = 0;
function toast(msg, ms = 2000) { const el = $("toast"); el.textContent = msg; el.classList.add("show"); clearTimeout(toastT); toastT = setTimeout(() => el.classList.remove("show"), ms); }

$("speed").onchange = (e) => { speed = +e.target.value; };
$("mind").onchange = (e) => { mind.visible = e.target.checked; $("probs").hidden = !e.target.checked; };
$("volume").oninput = (e) => sound.setVolume(+e.target.value);
$("mute").onclick = (e) => { const m = e.target.getAttribute("aria-pressed") !== "true"; e.target.setAttribute("aria-pressed", m); e.target.textContent = m ? "sound off" : "sound on"; sound.setMuted(m); };

// ------------------------------------------------------------------ frame loop
const clock = new THREE.Clock();
let frondT = 0, worldLineT = 0;
function frame() {
  const dt = Math.min(clock.getDelta(), 0.1);
  const t = clock.elapsedTime;
  groundU.uTime.value = t;
  if (world && !paused) {
    if (endTimer > 0) { endTimer -= dt; if (endTimer <= 0) newEpisode(); }
    else { acc += dt * speed; while (acc >= TICK) { acc -= TICK; tick(); if (endTimer > 0) break; } }
  }
  if (world && cur) render(dt, t, paused || endTimer > 0 ? 1 : clamp(acc / TICK, 0, 1));
  renderer.render(scene, cam);
  requestAnimationFrame(frame);
}

function render(dt, t, a) {
  const s = cur, p0 = prev || cur;
  const x = lerp(p0.x, s.x, a), y = lerp(p0.y, s.y, a), h = angLerp(p0.heading, s.heading, a);
  const light = lerp(p0.light, s.light, a);
  // light & atmosphere: day ↔ night, rain
  groundU.uLight.value = light; groundU.uRain.value = lerp(groundU.uRain.value, s.raining ? 1 : 0, dt * 0.5);
  sun.intensity = 0.55 + 1.6 * light; sun.color.setHSL(lerp(0.62, 0.09, light), lerp(0.35, 0.45, light), lerp(0.75, 0.86, light));
  hemi.intensity = 0.6 + 0.6 * light;
  const bg = new THREE.Color(pal.abyss).lerp(new THREE.Color(0x10303c), light * 0.6);
  scene.background = bg; scene.fog.color.copy(bg);
  // camera: dead zone + damping + small lookahead
  const vx = toX(x), vz = toZ(y);
  const dz = (cam.right - cam.left) * 0.15, dzz = (cam.top - cam.bottom) * 0.12;
  const look = reduceMotion ? 0 : clamp(s.v / C.v_max, 0, 1) * 0.65 * 0.7;
  const tx = vx + Math.cos(h) * look, tz = vz + Math.sin(h) * look;
  const want = camTarget.clone();
  if (tx - camTarget.x > dz) want.x = tx - dz; if (camTarget.x - tx > dz) want.x = tx + dz;
  if (tz - camTarget.z > dzz) want.z = tz - dzz; if (camTarget.z - tz > dzz) want.z = tz + dzz;
  camTarget.lerp(want, reduceMotion ? 1 : 1 - Math.exp(-dt / 0.32));
  camTarget.x = clamp(camTarget.x, -W / 2 + 3, W / 2 - 3); camTarget.z = clamp(camTarget.z, -H / 2 + 2, H / 2 - 2);
  placeCamera();
  sun.position.set(camTarget.x + 10, 18, camTarget.z - 8); sun.target.position.copy(camTarget);
  // Vela: root follows the simulation exactly; the child carries expressive secondary motion
  const yaw = yawOf(h);
  velaRoot.position.set(vx, 0.06, vz);
  if (vela) {
    const motion = s.resting || world.v < 0.02 ? "idle" : behave.closing >= 3 ? "nearingSeed" : "moving";
    // root transform lives on velaRoot; the controller only drives clips (local pose stays at the origin)
    vela.accept({ episode, sequence: s.t, x: 0, y: 0, z: 0, yaw: 0, motion, lowEnergy: s.energy < 0.2, events: velaEvents.splice(0) });
    vela.update(dt);
  }
  velaRoot.rotation.y = yaw;
  const threat = behave.threatT > 0.2;
  const breath = reduceMotion ? 0 : Math.sin(t * Math.PI * 2 / (threat ? 1.1 : s.resting ? 3.6 : 2.8)) * (s.resting ? 0.03 : 0.02);
  const crouch = s.resting ? 0.82 : s.inShelter && threat ? 0.78 : threat && s.v < 0.07 ? 0.88 : 1;
  velaChild.scale.set(1 + breath * 0.5, lerp(velaChild.scale.y, crouch, dt * 4) + breath, 1 + breath * 0.5);
  velaChild.rotation.z = lerp(velaChild.rotation.z, clamp(-s.w * 2.0, -0.21, 0.21), dt * 6);  // bank into turns (≤12°)
  velaChild.rotation.x = lerp(velaChild.rotation.x, s.dashing ? -0.12 : s.v < 0.02 ? 0.04 : 0, dt * 5);
  const tremble = s.inShelter && threat && !reduceMotion ? 0.006 : 0;
  velaChild.position.set((Math.random() - 0.5) * tremble, 0.03 + (s.inWater ? -0.05 : 0), (Math.random() - 0.5) * tremble);
  velaShadow.position.set(vx, 0.012, vz); velaShadow.material.opacity = s.inWater ? 0.2 : 0.75;
  velaGlow.position.set(vx, 0.6, vz); velaGlow.intensity = (0.25 + 0.9 * (1 - light)) * (0.5 + 0.5 * s.energy) * (s.inShelter ? 0.4 : 1);
  velaHalo.material.opacity = (0.14 + 0.22 * (1 - light)) * (s.inShelter ? 0.4 : 1);
  // seeds
  view.seeds.forEach((sd, i) => {
    const on = s.seedOn[i];
    const target = on ? 1 : 0;
    sd.s = lerp(sd.s, target, dt * (on ? 1.6 : 6));
    if (!on && sd.s < 0.02) sd.g.visible = false; else sd.g.visible = true;
    if (on) sd.g.position.set(toX(s.seedPos[i][0]), 0, toZ(s.seedPos[i][1]));
    sd.g.scale.setScalar(Math.max(sd.s, 0.001));
    sd.core.position.y = 0.16 + (reduceMotion ? 0 : Math.sin(t * 1.3 + sd.ph) * 0.025);
    sd.core.rotation.y = t * 0.4 + sd.ph;
  });
  // vents
  world.vents.forEach((v, i) => { const vv = view.vents[i]; if (!vv) return; vv.on = lerp(vv.on, v.on ? 1 : 0, dt * 2);
    vv.glow.material.opacity = vv.on * (0.18 + 0.08 * Math.sin(t * 3.1 + i));
    vv.segs.forEach((sg, k) => { sg.material.emissiveIntensity = 0.15 + vv.on * (1.1 + 0.4 * Math.sin(t * 2 + k)); sg.position.y = 0.04 + vv.on * 0.03 * Math.sin(t * 4 + k); }); });
  // thickets: wind + parting around Vela and the predator
  frondT += dt;
  const fr = view.fronds, m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), sc = new THREE.Vector3(), ps = new THREE.Vector3();
  const px = toX(world.pred.x), pz = toZ(world.pred.y);
  for (let i = 0; i < fr.inst.count; i++) {
    const b = fr.base[i];
    const dv = Math.hypot(b.x - vx, b.z - vz), dp = Math.hypot(b.x - px, b.z - pz);
    const push = Math.max(0, 1 - dv / 0.7) * 0.9 + Math.max(0, 1 - dp / 0.9) * 0.6;
    const wind = reduceMotion ? 0 : Math.sin(frondT * (1.1 + 0.5 * s.raining) + b.ph + b.x * 0.4) * (0.08 + 0.1 * s.raining);
    e.set(b.lean + wind + push, b.yaw, 0, "YXZ");
    if (push > 0) { const ang = Math.atan2(b.z - vz, b.x - vx); e.set(b.lean + wind + push, -ang + Math.PI / 2, 0, "YXZ"); }
    q.setFromEuler(e); sc.set(1, b.hgt * (1 - 0.15 * push), 1); ps.set(b.x, 0, b.z);
    m4.compose(ps, q, sc); fr.inst.setMatrixAt(i, m4);
  }
  fr.inst.instanceMatrix.needsUpdate = true;
  frondMat.emissiveIntensity = 0.2 + 0.4 * (1 - light);
  // predator body: head follows the simulation; segments trail along its real path
  const P0 = p0.pred, P1 = s.pred;
  shade.visible = P1.on;
  if (P1.on) {
    const hx = toX(lerp(P0.x, P1.x, a)), hz = toZ(lerp(P0.y, P1.y, a));
    const last = shadeTrail[0];
    if (!last || Math.hypot(last[0] - hx, last[1] - hz) > 0.04) { shadeTrail.unshift([hx, hz]); if (shadeTrail.length > 120) shadeTrail.pop(); }
    const stalking = P1.mode === "hunt" && !P1.pounce;
    const low = stalking ? 0.55 : P1.mode === "rest" ? 0.6 : 1;
    let distAcc = 0, ti = 0;
    for (let k = 0; k < SEG; k++) {
      const want = k * 0.24;
      while (ti < shadeTrail.length - 1 && distAcc + Math.hypot(shadeTrail[ti + 1][0] - shadeTrail[ti][0], shadeTrail[ti + 1][1] - shadeTrail[ti][1]) < want) {
        distAcc += Math.hypot(shadeTrail[ti + 1][0] - shadeTrail[ti][0], shadeTrail[ti + 1][1] - shadeTrail[ti][1]); ti++; }
      const pt = shadeTrail[Math.min(ti, shadeTrail.length - 1)] || [hx, hz];
      const sg = shadeSegs[k];
      const undul = reduceMotion ? 0 : Math.sin(t * (P1.pounce ? 9 : 3) - k * 0.8) * 0.02 * k;
      sg.m.position.set(pt[0] + undul, sg.r * 0.62 * low + 0.04 + (k === 0 ? 0.03 : 0), pt[1] + undul);
      const nx = shadeTrail[Math.min(ti + 1, shadeTrail.length - 1)] || pt;
      sg.m.rotation.y = Math.atan2(pt[0] - nx[0], pt[1] - nx[1]);
    }
    const head = shadeSegs[0].m;
    eyes.forEach((ey) => { const yawh = head.rotation.y, off = 0.11 * ey.userData.sd; ey.position.set(head.position.x + Math.sin(yawh) * 0.25 + Math.cos(yawh) * off, head.position.y + 0.07, head.position.z + Math.cos(yawh) * 0.25 - Math.sin(yawh) * off); });
    eyeMat.opacity = P1.mode === "hunt" ? (P1.pounce ? 1 : 0.55) : P1.mode === "search" ? 0.4 : P1.mode === "rest" ? 0.05 : 0.18;
    eyeMat.color.setHex(P1.pounce ? 0xfff6d0 : 0xd8e6b8);
    if (shadeMat.sheenColor) shadeMat.sheenColor.setHSL(0.72, 0.5, 0.18 + 0.12 * (1 - light));
    shadeShadow.position.set(hx, 0.012, hz); shadeShadow.material.opacity = 0.65;
  }
  // rain + motes
  rainMat.opacity = lerp(rainMat.opacity, s.raining ? 0.35 : 0, dt * 0.8);
  if (rainMat.opacity > 0.01) {
    for (let i = 0; i < RAIN_N; i++) { const r = rainSeeds[i]; r[1] -= dt * 9; if (r[1] < 0) { r[1] += 12; r[0] = Math.random() * 30 - 15; r[2] = Math.random() * 30 - 15; }
      const X = camTarget.x + r[0], Z = camTarget.z + r[2]; rainPos.set([X, r[1], Z, X + 0.03, r[1] + 0.35, Z], i * 6); }
    rainGeo.attributes.position.needsUpdate = true;
  }
  for (let i = 0; i < MOTE_N; i++) { const m = moteSeeds[i]; const tt = reduceMotion ? 0 : t;
    motePos.set([camTarget.x + m[0] + Math.sin(tt * 0.1 + m[3]) * 0.6, m[1] + Math.sin(tt * 0.3 + m[3]) * 0.2, camTarget.z + m[2] + Math.cos(tt * 0.08 + m[3]) * 0.6], i * 3); }
  moteGeo.attributes.position.needsUpdate = true;
  motes.material.opacity = 0.15 + 0.45 * light * (s.raining ? 0.3 : 1);
  // mind view
  if (mind.visible && s.obs) {
    for (let k = 0; k < C.n_rays; k++) {
      const ang = h + RAY_REL[k]; const o = s.obs; const b = N_CH * k;
      const ch = [o[b], o[b + 1], o[b + 2], o[b + 3], o[b + 4], o[b + 5]];
      const strongest = ch.indexOf(Math.max(...ch)); const prox = ch[strongest];
      const len = prox > 0 ? 6 * (1 - prox) : 6;
      const col = prox <= 0 ? [0.2, 0.3, 0.35] : [[0.5, 0.6, 0.65], [1, 0.7, 0.42], [0.95, 0.59, 0.54], [0.27, 0.9, 0.84], [1, 0.25, 0.3], [0.4, 0.6, 1]][strongest];
      mindPos.set([vx, 0.08, vz, vx + Math.cos(ang) * len, 0.08, vz + Math.sin(ang) * len], k * 6);
      mindCol.set([...col, ...col], k * 6);
    }
    mindGeo.attributes.position.needsUpdate = true; mindGeo.attributes.color.needsUpdate = true;
    drawProbs(s.probs, s.action);
  }
  // HUD
  $("energyBar").style.width = `${(s.energy * 100).toFixed(1)}%`;
  $("staminaBar").style.width = `${(s.stamina * 100).toFixed(1)}%`;
  worldLineT -= dt;
  if (worldLineT <= 0) { worldLineT = 0.5;
    const tod = ((world.t + world.t0) % C.day_ticks) / C.day_ticks, hrs = (tod * 24 + 12) % 24;  // light peaks at tod 0 = noon
    const hh = Math.floor(hrs), mm = Math.floor((hrs - hh) * 60);
    $("worldLine").textContent = `day ${(world.t / C.day_ticks + 1).toFixed(0)} · ${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")} · ${light > 0.6 ? "day" : light > 0.25 ? "dusk" : "night"}${s.raining ? " · rain" : ""} · seeds eaten ${s.epFood}`; }
  sound.update({ ...s, x: lerp(p0.x, s.x, a), y: lerp(p0.y, s.y, a) }, dt);
}

const ACTION_LABELS = ["cruise", "veer left", "veer right", "dash", "turn left", "turn right", "creep", "rest"];
function drawProbs(probs, chosen) {
  const el = $("probs"); if (!probs) { el.textContent = "untrained brain: no probabilities"; return; }
  if (!el.childElementCount) for (let i = 0; i < 8; i++) { const a = document.createElement("span"); a.textContent = ACTION_LABELS[i]; const b = document.createElement("span"); b.appendChild(document.createElement("i")); el.append(a, b); }
  for (let i = 0; i < 8; i++) { const lab = el.children[2 * i], bar = el.children[2 * i + 1].firstChild; lab.className = i === chosen ? "chosen" : ""; bar.style.width = `${(probs[i] * 100).toFixed(0)}%`; }
}

// ------------------------------------------------------------------ boot
async function boot() {
  try {
    await loadBrains();
    const gltf = await new GLTFLoader().loadAsync("public/assets/Vela.glb");
    vela = createVelaController(THREE, gltf);
    const scale = (C.creature_radius * 2) / 0.74 * 1.05;  // span 0.74 body lengths → creature diameter
    vela.object.scale.setScalar(scale);
    vela.object.traverse((o) => {
      if (!o.isMesh || !o.material) return;
      if (lite && o.material.isMeshPhysicalMaterial) {
        const m = o.material, std = new THREE.MeshStandardMaterial({ color: m.color, map: m.map, normalMap: m.normalMap, roughness: m.roughness,
          metalness: m.metalness, emissive: m.emissive, emissiveMap: m.emissiveMap, emissiveIntensity: m.emissiveIntensity });
        o.material = std;
      }
    });
    velaChild.add(vela.object);
    newEpisode();
    $("loadNote").textContent = Object.keys(policies).length ? "ready" : "ready (no trained policy found; Vela will stay still)";
    requestAnimationFrame(frame);
  } catch (err) {
    console.error(err);
    $("loadNote").textContent = `Could not load the Fold: ${err.message}`;
  }
}
$("enter").onclick = async () => { $("veil").classList.add("gone"); try { await sound.start(); sound.setVolume(+$("volume").value); } catch (e) { toast("Sound is unavailable in this browser.", 3000); } };
if (params.has("auto")) $("veil").classList.add("gone");  // screenshots / kiosk
boot();
