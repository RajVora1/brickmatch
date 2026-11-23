"""Interactive 3D viewer for a build: drag to turn it, scroll to zoom, missing pieces in red.

It reads the same `layout` as the step-by-step instructions (see docs/build-instructions.md) and makes a small
web page that draws the build with three.js (a WebGL library). WebGL has a depth buffer: for every pixel it keeps
whichever surface is nearest the camera, so pieces never look like they are inside each other (the problem with
matplotlib's 3D mode, which sorts whole faces instead).

Each piece is built here in Python as a closed solid (a box, or a wedge-shaped prism for slopes) shrunk by a hair
on every side, so neighbouring pieces show a thin seam instead of fighting over the same surface. The page only
turns those solids into triangles, so all the geometry can be tested without a browser.

Use it from Python (in Colab the page shows right in the notebook):

    from brickmatch.viewer import show
    show("house_hut", have={"brick_2x4": 4, "brick_1x2": 6})

Or save a page you can open in any browser:

    python -m brickmatch.viewer house_hut --have '{"brick_2x4": 4}' --out hut_3d.html

The page loads three.js from the jsDelivr CDN, so it needs an internet connection.
"""
import argparse
import html
import json
from pathlib import Path

from brickmatch.instructions import NAMES, SLOPES, _rgb, cells, footprint, get_build, make_steps, stud_cells

PLATE = 0.4  # plate height in studs (a real plate is 3.2 mm, a stud pitch 8 mm)
GAP = 0.012  # how far each side of a piece is pulled in, so neighbours show a seam
STUD_R, STUD_H = 0.3, 0.17
MISSING_RGB = (1.0, 0.35, 0.35)  # see-through and pulsing in the page
THREE_VERSION = "0.160.0"


def _solid(p):
    """A placed piece as a closed solid: (vertices, faces). Vertices are (x, y, z) in studs with z up;
    each face is a list of vertex indices around one flat side.
    """
    w, d, h = footprint(p)
    if p["piece"] in SLOPES:
        sw = (w if p.get("rot", 0) in (0, 180) else d)  # width across the slope at rot 0 (1 or 2)
        # Side view at rot 0 (y, z): front bottom, back bottom, back top, ridge (end of the stud row), front top.
        # The slope runs from the full height at y=1 down to 1 plate at the front edge (y=0).
        profile = [(0, 0), (2, 0), (2, 3 * PLATE), (1, 3 * PLATE), (0, PLATE)]
        verts = [(x, y, z) for x in (0, sw) for y, z in profile]
        n = len(profile)
        faces = [list(range(n)), list(range(n, 2 * n))]  # the two pentagon ends
        faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]  # bottom, back, top, slope, front
        verts = [_turn(v, p.get("rot", 0)) for v in verts]
    else:
        verts = [(x, y, z) for x in (0, w) for y in (0, d) for z in (0, h * PLATE)]
        # Vertex index = 4*xi + 2*yi + zi.
        faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    # Move so the piece's front-left-bottom corner sits on its layout spot.
    mx, my = min(v[0] for v in verts), min(v[1] for v in verts)
    verts = [(v[0] - mx + p["x"], v[1] - my + p["y"], v[2] + p["z"] * PLATE) for v in verts]
    return _shrink(verts), faces


def _turn(v, rot):
    """Turn a point counterclockwise (seen from above) by 0, 90, 180 or 270 degrees around the z axis."""
    x, y, z = v
    for _ in range(rot // 90):
        x, y = -y, x
    return x, y, z


def _shrink(verts):
    """Pull the outer sides of a piece in by GAP (top and bottom stay put, so stacked pieces still touch)."""
    x0, x1 = min(v[0] for v in verts), max(v[0] for v in verts)
    y0, y1 = min(v[1] for v in verts), max(v[1] for v in verts)

    def pull(a, lo, hi):
        return a + GAP if abs(a - lo) < 1e-9 else a - GAP if abs(a - hi) < 1e-9 else a
    return [(round(pull(x, x0, x1), 4), round(pull(y, y0, y1), 4), round(z, 4)) for x, y, z in verts]


def _hex(rgb):
    return "#" + "".join(f"{round(c * 255):02x}" for c in rgb)


def scene(build, have=None):
    """Everything the page needs to draw a build, as plain data (it becomes JSON in the page).

    have: piece counts you own (e.g. from the detector). Pieces are used up in step order, the same way as
    the step-by-step instructions, so the pieces marked missing here match the red pieces in the booklet.
    """
    steps = make_steps(build, have)
    filled = set().union(*(cells(p) for p in build["layout"]))
    pieces = []
    for s in steps:
        for p in s["pieces"]:
            verts, faces = _solid(p)
            top = p["z"] + footprint(p)[2]
            studs = [[sx + 0.5, sy + 0.5, round(top * PLATE, 4)] for sx, sy in sorted(stud_cells(p))
                     if (sx, sy, top) not in filled]  # skip studs hidden inside the piece above
            missing = p["owned"] is False
            pieces.append({
                "name": NAMES.get(p["piece"], p["piece"]),
                "step": s["number"],
                "n": p["n"],
                "where": p["where"],
                "missing": missing,
                "color": _hex(MISSING_RGB if missing else _rgb(p)),
                "verts": verts,
                "faces": faces,
                "studs": studs,
            })
    xs = [v[0] for p in pieces for v in p["verts"]]
    ys = [v[1] for p in pieces for v in p["verts"]]
    zs = [v[2] for p in pieces for v in p["verts"]]
    return {
        "name": build["name"],
        "steps": len(steps),
        "checked": have is not None,
        "missing": sum(p["missing"] for p in pieces),
        "total": len(pieces),
        "center": [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2],
        "size": max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)),
        "floor": [min(xs), max(xs), min(ys), max(ys)],
        "stud": [STUD_R, STUD_H],
        "pieces": pieces,
    }


def page(build, have=None, height=480):
    """A complete web page (HTML) with the interactive 3D view of one build."""
    data = json.dumps(scene(build, have), separators=(",", ":"))
    return (_PAGE.replace("__DATA__", data.replace("</", "<\\/"))
            .replace("__THREE__", THREE_VERSION).replace("__HEIGHT__", str(height)))


def embed(build, have=None, height=480):
    """The page wrapped in an <iframe>, ready for gr.HTML or a notebook. The iframe keeps the page's script
    and styles separate from whatever page it sits in.
    """
    doc = html.escape(page(build, have, height), quote=True)
    return (f'<iframe srcdoc="{doc}" style="width:100%;height:{height + 4}px;border:0;border-radius:8px" '
            f'title="3D view of {html.escape(build["name"])}"></iframe>')


def show(build_id, have=None, height=480):
    """Show the 3D view in a Jupyter or Colab notebook."""
    from IPython.display import HTML  # noqa: PLC0415
    return HTML(embed(get_build(build_id), have, height))


def save(build, have, path, height=640):
    path = Path(path)
    path.write_text(page(build, have, height), encoding="utf-8")
    return path


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D view</title>
<style>
  :root { --bg:#f4f5f7; --panel:rgba(255,255,255,.92); --ink:#1d2129; --muted:#5b6270; --line:#d6d9de; --accent:#d62828; }
  html,body { margin:0; height:100%; background:var(--bg); font:14px/1.35 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; color:var(--ink); }
  #wrap { position:relative; width:100%; height:100vh; min-height:__HEIGHT__px; overflow:hidden; }
  canvas { display:block; touch-action:none; cursor:grab; }
  canvas:active { cursor:grabbing; }
  .panel { position:absolute; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:8px 10px; }
  #title { top:10px; left:10px; max-width:calc(100% - 40px); }
  #title b { font-size:15px; }
  #title .sub { color:var(--muted); margin-top:2px; }
  #title .miss { color:var(--accent); font-weight:600; }
  #controls { left:10px; right:10px; bottom:10px; display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; }
  #controls label { display:flex; align-items:center; gap:6px; white-space:nowrap; }
  #controls input[type=range] { width:min(220px,40vw); }
  #controls button { font:inherit; padding:4px 10px; border:1px solid var(--line); border-radius:6px; background:#fff; cursor:pointer; }
  #tip { position:absolute; pointer-events:none; display:none; background:#1d2129; color:#fff; padding:5px 8px; border-radius:6px; font-size:12px; max-width:260px; }
  #tip .m { color:#ff9a9a; font-weight:600; }
  #err { position:absolute; inset:0; display:none; align-items:center; justify-content:center; text-align:center; padding:20px; color:var(--muted); }
  .hint { color:var(--muted); font-size:12px; }
</style></head>
<body><div id="wrap">
  <div id="title" class="panel"><b id="name"></b><div class="sub" id="sub"></div></div>
  <div id="controls" class="panel">
    <label>Steps <input id="step" type="range" min="1" value="1"> <span id="steplabel"></span></label>
    <label id="misswrap"><input id="showmiss" type="checkbox" checked> Missing pieces</label>
    <label><input id="spin" type="checkbox"> Spin</label>
    <button id="reset" type="button">Reset view</button>
    <span class="hint">Drag to turn, scroll or pinch to zoom, right-drag to move</span>
  </div>
  <div id="tip"></div>
  <div id="err">The 3D view needs an internet connection to load three.js.</div>
</div>
<script type="module">
const S = __DATA__;
const wrap = document.getElementById('wrap');
let THREE, OrbitControls;
try {
  THREE = await import('https://cdn.jsdelivr.net/npm/three@__THREE__/+esm');
  ({ OrbitControls } = await import('https://cdn.jsdelivr.net/npm/three@__THREE__/examples/jsm/controls/OrbitControls.js/+esm'));
} catch (e) {
  document.getElementById('err').style.display = 'flex';
  throw e;
}

// ----- header and controls -----
document.getElementById('name').textContent = S.name;
const sub = document.getElementById('sub');
if (!S.checked) sub.textContent = `${S.total} pieces, ${S.steps} steps`;
else if (S.missing === 0) sub.textContent = `${S.total} pieces, ${S.steps} steps. You have every piece.`;
else sub.innerHTML = `${S.total} pieces, ${S.steps} steps. <span class="miss">${S.missing} missing</span> (shown in red, see-through)`;
if (!S.missing) document.getElementById('misswrap').style.display = 'none';
const slider = document.getElementById('step');
slider.max = S.steps; slider.value = S.steps;
if (S.steps < 2) slider.parentElement.style.display = 'none';

// ----- scene -----
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0xf4f5f7);
wrap.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 0.05, 500);
camera.up.set(0, 0, 1);  // z is up, like the layout
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(-0.4, -0.9, 1.3);
scene.add(sun);
const fill = new THREE.DirectionalLight(0xffffff, 0.5);
fill.position.set(1, 0.6, 0.4);
scene.add(fill);

const [cx, cy, cz] = S.center;
const floorZ = -0.002;
{ // stud grid on the table, a bit bigger than the build
  const [x0, x1, y0, y1] = S.floor;
  const gx0 = Math.floor(x0) - 2, gx1 = Math.ceil(x1) + 2, gy0 = Math.floor(y0) - 2, gy1 = Math.ceil(y1) + 2;
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(gx1 - gx0, gy1 - gy0),
    new THREE.MeshBasicMaterial({ color: 0xe9ebee }));
  plane.position.set((gx0 + gx1) / 2, (gy0 + gy1) / 2, floorZ - 0.001);
  scene.add(plane);
  const pts = [];
  for (let x = gx0; x <= gx1; x++) pts.push(x, gy0, floorZ, x, gy1, floorZ);
  for (let y = gy0; y <= gy1; y++) pts.push(gx0, y, floorZ, gx1, y, floorZ);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  scene.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xd3d6db })));
}

// A piece's flat sides -> triangles. Each side is turned to face outward (away from the piece's middle).
function solidGeometry(verts, faces) {
  const c = verts.reduce((a, v) => [a[0] + v[0] / verts.length, a[1] + v[1] / verts.length, a[2] + v[2] / verts.length], [0, 0, 0]);
  const pos = [];
  for (let f of faces) {
    const P = f.map(i => new THREE.Vector3(...verts[i]));
    const n = new THREE.Vector3().crossVectors(P[1].clone().sub(P[0]), P[2].clone().sub(P[0]));
    const mid = P.reduce((a, p) => a.add(p), new THREE.Vector3()).multiplyScalar(1 / P.length);
    if (n.dot(mid.sub(new THREE.Vector3(...c))) < 0) P.reverse();
    for (let i = 1; i < P.length - 1; i++) for (const q of [P[0], P[i], P[i + 1]]) pos.push(q.x, q.y, q.z);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.computeVertexNormals();  // not indexed, so each triangle keeps its own flat normal
  return g;
}

const studGeo = new THREE.CylinderGeometry(S.stud[0], S.stud[0], S.stud[1], 24, 1, false);
studGeo.rotateX(Math.PI / 2);
studGeo.translate(0, 0, S.stud[1] / 2);
const studEdges = new THREE.EdgesGeometry(studGeo, 30);
const pickable = [], ghosts = [];
const groups = S.pieces.map(p => {
  const color = new THREE.Color(p.color);
  const ghost = p.missing;
  const mat = new THREE.MeshStandardMaterial({
    color, roughness: 0.55, metalness: 0.0, flatShading: false,
    transparent: ghost, opacity: ghost ? 0.35 : 1, depthWrite: !ghost,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,  // keep outlines on top of their faces
  });
  // Outlines: dark for owned pieces (like the booklet), bright red for missing ones.
  const edgeMat = new THREE.LineBasicMaterial({ color: ghost ? 0xe00000 : new THREE.Color(p.color).multiplyScalar(0.08) });
  const grp = new THREE.Group();
  const body = solidGeometry(p.verts, p.faces);
  const mesh = new THREE.Mesh(body, mat);
  mesh.userData = p;
  pickable.push(mesh);
  grp.add(mesh, new THREE.LineSegments(new THREE.EdgesGeometry(body, 20), edgeMat));
  for (const [x, y, z] of p.studs) {
    const s = new THREE.Mesh(studGeo, mat);
    s.position.set(x, y, z);
    s.userData = p;
    pickable.push(s);
    const e = new THREE.LineSegments(studEdges, edgeMat);
    e.position.set(x, y, z);
    grp.add(s, e);
  }
  if (ghost) { grp.renderOrder = 1; ghosts.push(mat); }  // see-through pieces are drawn after the solid ones
  scene.add(grp);
  return grp;
});

// ----- camera and controls -----
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.maxPolarAngle = Math.PI * 0.495;  // don't go under the table
function resetView() {
  const r = Math.max(S.size, 4) * 2.3;
  // From the front right and above, like the step pictures.
  camera.position.set(cx + r * 0.55, cy - r * 0.95, cz + r * 0.7);
  controls.target.set(cx, cy, cz * 0.9);
  controls.minDistance = r * 0.25; controls.maxDistance = r * 4;
  controls.update();
}
resetView();

function applyFilters() {
  const upTo = +slider.value, showMiss = document.getElementById('showmiss').checked;
  S.pieces.forEach((p, i) => { groups[i].visible = p.step <= upTo && (showMiss || !p.missing); });
  document.getElementById('steplabel').textContent = upTo === S.steps ? `all ${S.steps}` : `1-${upTo} of ${S.steps}`;
}
slider.addEventListener('input', applyFilters);
document.getElementById('showmiss').addEventListener('change', applyFilters);
document.getElementById('spin').addEventListener('change', e => { controls.autoRotate = e.target.checked; });
controls.autoRotateSpeed = 2.0;
document.getElementById('reset').addEventListener('click', resetView);
applyFilters();

// ----- hover: say which piece is under the pointer -----
const tip = document.getElementById('tip');
const ray = new THREE.Raycaster();
renderer.domElement.addEventListener('pointermove', ev => {
  const r = renderer.domElement.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2((ev.clientX - r.left) / r.width * 2 - 1, -(ev.clientY - r.top) / r.height * 2 + 1), camera);
  const hit = ray.intersectObjects(pickable.filter(m => m.parent.visible), false)[0];
  if (!hit) { tip.style.display = 'none'; return; }
  const p = hit.object.userData;
  tip.innerHTML = `<b>${p.name}</b>${p.missing ? ' <span class="m">missing</span>' : ''}<br>Step ${p.step}, piece ${p.n}: ${p.where}`;
  tip.style.display = 'block';
  tip.style.left = Math.min(ev.clientX - r.left + 14, r.width - 270) + 'px';
  tip.style.top = (ev.clientY - r.top + 14) + 'px';
});
renderer.domElement.addEventListener('pointerleave', () => { tip.style.display = 'none'; });

function resize() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = w + 'px'; renderer.domElement.style.height = h + 'px';
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(wrap);
resize();
const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  // Missing pieces pulse gently so they stand out even next to red bricks you own.
  const a = 0.22 + 0.16 * (0.5 + 0.5 * Math.sin(clock.getElapsedTime() * 3));
  for (const m of ghosts) m.opacity = a;
  controls.update();
  renderer.render(scene, camera);
});
window.__view = { camera, controls, resetView };
window.__ready = true;
</script></body></html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Save an interactive 3D view of a build as a web page.")
    ap.add_argument("build", help="build id, e.g. house_hut")
    ap.add_argument("--have", help="JSON piece counts you own; missing pieces are shown in red")
    ap.add_argument("--out", help="where to save the page (default: <build>_3d.html)")
    args = ap.parse_args()
    b = get_build(args.build)
    if not b.get("layout"):
        raise SystemExit(f"{b['name']} has no layout yet, so it can't be shown in 3D.")
    print("Saved", save(b, json.loads(args.have) if args.have else None, args.out or f"{b['id']}_3d.html"))
