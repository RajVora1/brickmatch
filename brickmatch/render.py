"""Draw LEGO builds as clean pictures with no pieces poking through each other.

matplotlib's 3D mode sorts whole faces by their average depth, which goes wrong when faces of different sizes
are close together (that's why pieces looked like they were inside each other). This renderer avoids that by
cutting every piece into unit cells (one stud square by one plate high) and painting the cells from the back to
the front (the "painter's algorithm"). On a grid of equal cells that order is always correct: a cell can only
hide a cell that is further away, and every cell further away is painted first.

Slopes are cut the same way: the downhill row of a slope is split into plate-high slices, and each slice is
trimmed by the sloped surface.

Coordinates: x and y in studs, height in plates (like the layout format); a plate is drawn 0.4 studs tall.
The camera looks from the front right and above, so you see each piece's top, front (low y) and right side (high x).
"""
import math

from brickmatch.instructions import SLOPE_FACING, SLOPES, _rgb, footprint, stud_cells

PLATE = 0.4  # plate height in studs
STUD_R, STUD_H = 0.3, 0.17
THETA, PHI = math.radians(32), math.radians(30)  # camera turn and tilt
CAM = (math.sin(THETA) * math.cos(PHI), -math.cos(THETA) * math.cos(PHI), math.sin(PHI))  # toward the camera
LIGHT = (0.15, -0.6, 0.78)
EDGE = (0.12, 0.12, 0.12)
MISSING_EDGE = (0.85, 0.05, 0.05)


def project(x, y, z):
    """3D point (studs, studs, studs) -> 2D picture point."""
    sx = x * math.cos(THETA) + y * math.sin(THETA)
    depth = x * math.sin(THETA) - y * math.cos(THETA)
    return sx, z * math.cos(PHI) - depth * math.sin(PHI)


def _dot(a, b):
    return sum(i * j for i, j in zip(a, b))


def _shade(rgb, normal):
    n = math.sqrt(_dot(normal, normal))
    lit = max(0.0, _dot(normal, LIGHT) / (n * math.sqrt(_dot(LIGHT, LIGHT))))
    k = 0.5 + 0.5 * lit
    return tuple(min(1.0, c * k) for c in rgb)


def _clip(poly, f):
    """Keep the part of a convex polygon where f(point) >= 0 (Sutherland-Hodgman)."""
    out = []
    for i, a in enumerate(poly):
        b = poly[(i + 1) % len(poly)]
        fa, fb = f(a), f(b)
        if fa >= -1e-9:
            out.append(a)
        if (fa >= -1e-9) != (fb >= -1e-9):
            t = fa / (fa - fb)
            out.append(tuple(pa + t * (pb - pa) for pa, pb in zip(a, b)))
    return out


def _area2d(pts):
    p = [project(*q) for q in pts]
    return abs(sum(p[i][0] * p[i - 1][1] - p[i - 1][0] * p[i][1] for i in range(len(p)))) / 2


BOX_FACES = {  # normal -> corners of that face of the unit box [0,1]^3 (z scaled later)
    (0, 0, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    (0, 0, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    (0, -1, 0): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    (0, 1, 0): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
    (-1, 0, 0): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
    (1, 0, 0): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
}


def _cells_of(p):
    """Yield (cell, kind) for every unit cell of a piece. kind is 'full' or ('slope', level, facing)."""
    w, d, h = footprint(p)
    studs = stud_cells(p)
    for i in range(w):
        for j in range(d):
            cx, cy = p["x"] + i, p["y"] + j
            for k in range(h):
                cell = (cx, cy, p["z"] + k)
                if p["piece"] in SLOPES and (cx, cy) not in studs and k > 0:
                    yield cell, ("slope", k, SLOPE_FACING[p.get("rot", 0)])
                else:
                    yield cell, "full"


def _slope_keep(cell, level, facing):
    """f(point) >= 0 inside the slope: the surface rises from 1 plate at the downhill edge to 3 at the back."""
    cx, cy, cz = cell
    z0 = cz - level
    fx, fy = facing

    def f(pt):
        x, y, z = pt
        if fx > 0:
            u = cx + 1 - x
        elif fx < 0:
            u = x - cx
        elif fy > 0:
            u = cy + 1 - y
        else:
            u = y - cy
        return (z0 + 1 + 2 * u) * PLATE - z
    return f


def _slope_normal(facing):
    fx, fy = facing
    return (fx * 2 * PLATE, fy * 2 * PLATE, 1.0)  # rises 2 plates over 1 stud, away from the facing direction


def render(ax, pieces, missing=frozenset(), studs=True, edge_width=1.0):
    """Draw pieces on a 2D matplotlib axis. `missing` holds id()s of pieces to draw see-through with red edges.

    Returns the 2D bounding box (xmin, xmax, ymin, ymax) of what was drawn.
    """
    from matplotlib.collections import PolyCollection  # noqa: PLC0415

    owner, full = {}, set()
    for p in pieces:
        for cell, kind in _cells_of(p):
            owner[cell] = (p, kind)
            if kind == "full":
                full.add(cell)

    items = []  # (paint order key, polygons...) appended in order later

    def emit(key, poly2d, fill, edge, lw):
        items.append((key, len(items), poly2d, fill, edge, lw))

    # Collect visible faces per cell, then work out which edges are piece outlines.
    faces = []  # (key, piece, plane, pts3d, rgb, alpha)
    for cell, (p, kind) in owner.items():
        cx, cy, cz = cell
        rgb = _rgb(p)
        alpha = 0.35 if id(p) in missing else 1.0
        key = cx - cy + cz
        for normal, corners in BOX_FACES.items():
            if _dot(normal, CAM) <= 1e-9:
                continue  # facing away from the camera
            nb = (cx + normal[0], cy + normal[1], cz + normal[2])
            if nb in full:
                continue  # pressed against a solid cell: hidden
            if kind != "full" and normal == (0, 0, 1):
                continue  # inside a slope: the slice above (or nothing, at the top) covers it
            pts = [(cx + a, cy + b, (cz + c) * PLATE) for a, b, c in corners]
            if kind != "full":
                pts = _clip(pts, _slope_keep(cell, kind[1], kind[2]))
                if len(pts) < 3 or _area2d(pts) < 1e-6:
                    continue
            elif normal == (0, 0, 1) and nb in owner and owner[nb][0] is p:
                continue  # top of a slope's bottom slice, covered by the slice above
            faces.append((key, p, normal, pts, _shade(rgb, normal), alpha))
        if kind != "full":
            _, level, facing = kind
            n = _slope_normal(facing)
            if _dot(n, CAM) > 1e-9:
                f = _slope_keep(cell, level, facing)
                # The sloped surface over this cell, cut to the slice's height band.
                cap = [(cx + a, cy + b, f((cx + a, cy + b, 0))) for a, b in [(0, 0), (1, 0), (1, 1), (0, 1)]]
                cap = _clip(cap, lambda q: (cz + 1) * PLATE - q[2])
                cap = _clip(cap, lambda q: q[2] - cz * PLATE) if cap else cap
                if len(cap) >= 3 and _area2d(cap) > 1e-6:
                    faces.append((key, p, n, cap, _shade(rgb, n), alpha))

    # An edge shared by two faces of the same piece lying in the same plane is inside a surface: don't outline it.
    def ekey(a, b):
        ra, rb = tuple(round(v, 4) for v in a), tuple(round(v, 4) for v in b)
        return (ra, rb) if ra < rb else (rb, ra)

    def plane(normal, pts):
        n = math.sqrt(_dot(normal, normal))
        nn = tuple(round(v / n, 4) for v in normal)
        return nn, round(_dot(nn, pts[0]), 4)

    count = {}
    for _, p, normal, pts, *_ in faces:
        pl = plane(normal, pts)
        for i in range(len(pts)):
            k = (id(p), pl, ekey(pts[i], pts[(i + 1) % len(pts)]))
            count[k] = count.get(k, 0) + 1

    for key, p, normal, pts, rgb, alpha in faces:
        edge = MISSING_EDGE if id(p) in missing else EDGE
        poly = [project(*q) for q in pts]
        emit(key, poly, (*rgb, alpha), (*rgb, alpha), 1.0 if alpha == 1.0 else 0.0)  # fill, edged in its own color to hide seams
        pl = plane(normal, pts)
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            if count[(id(p), pl, ekey(a, b))] == 1 and math.dist(project(*a), project(*b)) > 1e-3:
                emit(key, [project(*a), project(*b)], "none", edge, 0.9 * edge_width)

    if studs:
        for p in pieces:
            rgb = _rgb(p)
            alpha = 0.35 if id(p) in missing else 1.0
            edge = MISSING_EDGE if id(p) in missing else EDGE
            top = p["z"] + footprint(p)[2]
            for sx, sy in stud_cells(p):
                if (sx, sy, top) in owner:
                    continue
                key = sx - sy + top  # painted with the (empty) cell the stud pokes into
                base = [(sx + 0.5 + STUD_R * math.cos(t), sy + 0.5 + STUD_R * math.sin(t))
                        for t in (2 * math.pi * k / 20 for k in range(20))]
                lower = [project(x, y, top * PLATE) for x, y in base]
                upper = [project(x, y, top * PLATE + STUD_H) for x, y in base]
                emit(key, _hull(lower + upper), (*_shade(rgb, (0.6, -0.8, 0)), alpha), edge, 0.6 * edge_width)
                emit(key, upper, (*_shade(rgb, (0, 0, 1)), alpha), edge, 0.6 * edge_width)

    items.sort(key=lambda t: (t[0], t[1]))
    ax.add_collection(PolyCollection([t[2] for t in items], facecolors=[t[3] for t in items],
                                     edgecolors=[t[4] for t in items], linewidths=[t[5] for t in items],
                                     closed=True, joinstyle="round", capstyle="round"))
    xs = [x for t in items for x, _ in t[2]]
    ys = [y for t in items for _, y in t[2]]
    return min(xs), max(xs), min(ys), max(ys)


def _hull(points):
    """Convex hull (monotone chain) of 2D points."""
    pts = sorted(set((round(x, 6), round(y, 6)) for x, y in points))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for pt in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    for pt in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)
    return lower[:-1] + upper[:-1]


def bounds(pieces):
    """2D bounding box of the pieces' pictures (with studs), for giving every page the same framing."""
    xs, ys = [], []
    for p in pieces:
        w, d, h = footprint(p)
        for a in (p["x"], p["x"] + w):
            for b in (p["y"], p["y"] + d):
                for c in (p["z"] * PLATE, (p["z"] + h) * PLATE + STUD_H):
                    x, y = project(a, b, c)
                    xs.append(x)
                    ys.append(y)
    return min(xs), max(xs), min(ys), max(ys)


def frame(ax, box, pad=0.3):
    x0, x1, y0, y1 = box
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
