# Step-by-step build instructions

Each build in `builds/builds.json` has a parts list (what you need). To get instructions, a build also needs a
`layout`: where every piece goes. The code turns that into one step per layer, draws a picture of each step, and
checks off the pieces the detector found. None of this is machine learning. Like the matcher, it is plain code
that runs on the model's output.

All 24 builds have layouts. To make the parts fit a real build, a few descriptions changed (the parts lists did
not): Wall with window is 6 rows with a 2 x 3 window, Long bridge has a small hump in the middle, Step bridge
ends in a small lookout, Tiny house has its door on the side, Garage has a small back window, Cottage is 8x4,
and the Two-story house's 2x4 floor sticks out one stud on each side.

## What a step looks like

Each step is one layer. The picture has two halves:

- **Left:** the build so far in 3D. Earlier steps are grey, this step's pieces are in color, and pieces you're
  missing are red and see-through.
- **Right:** a map of just this layer, seen from above, on a stud grid. Columns are numbered left to right and rows
  front to back. Grey squares are the studs you're building on. Slopes have an arrow pointing downhill.

Every new piece has a number, and the list below the picture uses the same numbers:

```
Step 2: 3 pieces, on top of step 1
  ✓ 1. Brick 1 x 2: left to right over columns 1-2, row 1 (front)
  ✓ 2. Brick 1 x 4: left to right over columns 3-6, row 1 (front)
  ✗ 3. Brick 1 x 2: left to right over columns 7-8, row 1 (front) (missing)
```

## The layout format

```json
{"piece": "brick_1x4", "x": 2, "y": 0, "z": 3, "rot": 0}
```

| Field | Meaning |
|---|---|
| `piece` | A tier-1 piece id from `configs/pieces.yaml` |
| `x`, `y` | The stud the piece's corner sits on, counted from 0. x runs left to right, y runs front to back. |
| `z` | Height in **plate units**. A plate is 1 tall and a brick or slope is 3, so bricks in row 2 have z=3, row 3 z=6. |
| `rot` | 0, 90, 180 or 270. Turning 90 or 270 swaps length and width (a 1x4 at rot 90 runs front to back). For slopes it sets which way the slope faces: 0 = down toward the front, 90 = right, 180 = back, 270 = left. |
| `color` | Optional, a color id from `configs/colors.yaml`. |

At rot 0, bricks and plates lie along x (a 1x4 covers x to x+3), and slopes are 2 deep along y with their studs
on the back row.

The 3D viewer will read this same format, so a layout written now is reused there.

## Adding a new build

1. Build it for real (or sketch it on squared paper), one row at a time from the bottom.
2. For each piece, write down its corner stud, its row height in plate units, and its rotation.
3. Add the build to `builds/builds.json` with its `parts` and a `"layout": [...]`.
4. Run the checks: `python -m pytest tests`. They fail if the layout doesn't match the parts list, if two pieces
   overlap, or if a piece floats (isn't pressed onto studs below it, and a slope only has studs on its back row).
5. Look at the pictures:

```
python -m brickmatch.instructions wall_long --out steps/
```

Add `--have '{"brick_1x4": 6, "brick_1x2": 2}'` to see which pieces get checked off and where you'd get stuck.

## In the demo

Under the build table there is a **Step-by-step instructions** section. After **Find my pieces**, it switches to
the best-matching build. Grey pieces come from earlier steps, colored pieces are added in this
step, and red see-through pieces are the ones you're missing. Pieces are used up in step order, so when you're
short, the later steps are the ones marked missing.

## Known limits

- Pictures are drawn with matplotlib from one fixed angle, without studs. The 3D viewer is the place to fix that.
- A step is one whole layer. A layer with many pieces (a house wall has up to 8) could be split into smaller steps.
- The checks catch overlaps and floating pieces, but not a build that falls apart into two loose halves.
- Colors in the pictures come from the piece type unless the layout gives one. The checklist matches by type
  only, the same as the build matcher.

## LEGO-style booklets

`python -m brickmatch.booklet house_hut --out booklets/` makes a PDF that looks like a printed LEGO booklet:

- A cover page with the finished model.
- Small steps of at most 4 pieces. Each step has a parts box with "2x" counts, a big step number, and the model
  drawn with studs at the same scale on every page.
- A parts list page at the end.

Add `--have '{...}'` to mark missing pieces in red. So far only the Hut has been checked page by page.

All the pictures (booklets and step pictures) are drawn by `brickmatch/render.py`. It cuts every piece into
stud-by-plate cells and paints them from back to front, so pieces never look like they're inside each other.
matplotlib's own 3D mode caused that problem, because it sorts whole faces by average depth.
