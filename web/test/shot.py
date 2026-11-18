"""Preview check: python web/test/shot.py <out_dir> [url]. One headless page, console errors + 3 screenshots + frame timing.
Run inside the memory guard: python -m genesis.jobguard -- python web/test/shot.py <dir>"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8791/index.html?auto"
errors = []
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--autoplay-policy=no-user-gesture-required"])
    for name, vp in (("desktop", {"width": 1280, "height": 768}), ("phone", {"width": 390, "height": 844})):
        pg = b.new_page(viewport=vp)
        pg.on("console", lambda m: errors.append(f"{name} {m.type}: {m.text}"[:300]) if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: errors.append(f"{name} pageerror: {e}"[:300]))
        pg.goto(url, wait_until="networkidle")
        pg.wait_for_timeout(6000)
        pg.screenshot(path=str(out / f"{name}-1.png"))
        if name == "desktop":
            pg.check("#mind")
            pg.wait_for_timeout(5000)
            pg.screenshot(path=str(out / f"{name}-mind.png"))
            ft = pg.evaluate("""() => new Promise(r => { let n = 0, t0 = performance.now(); function f() { if (++n < 60) requestAnimationFrame(f); else r((performance.now() - t0) / 60); } requestAnimationFrame(f); })""")
            state = pg.evaluate("() => [document.getElementById('stateLabel').textContent, document.getElementById('worldLine').textContent, document.getElementById('brainTag').textContent, document.getElementById('loadNote').textContent]")
            print(json.dumps({"frame_ms_swiftshader": round(ft, 1), "hud": state}))
        pg.close()
    b.close()
print("\n".join(errors[:20]) if errors else "no console errors")
