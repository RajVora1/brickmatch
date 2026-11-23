"""Checks on fixing the model's counts before matching (brickmatch/counts.py).

    python -m pytest tests/test_counts.py
"""
import pytest

from brickmatch.counts import NAMES, PIECES, changes, read_table, table
from brickmatch.match import rank_builds


def edit(rows, name, n):
    for r in rows:
        if r[0] == name:
            r[2] = n
    return rows


def test_table_lists_every_type_with_found_ones_first():
    rows = table({"brick_2x2": 3, "brick_2x4": 5}, {"brick_2x4": "red 5"})
    assert [r[0] for r in rows[:2]] == ["Brick 2 x 4", "Brick 2 x 2"]
    assert rows[0] == ["Brick 2 x 4", 5, 5, "red 5"]
    assert len(rows) == len(PIECES)
    assert {r[0] for r in rows} == set(NAMES.values())
    # Tier-1 types come before tier-2 ones among the types the model didn't find.
    tiers = {p["name"]: p["tier"] for p in PIECES}
    rest = [tiers[r[0]] for r in rows[2:]]
    assert rest == sorted(rest)


def test_unedited_table_reads_back_as_the_model_counts():
    found = {"brick_2x2": 3, "brick_2x4": 5, "slope45_2x1": 2}
    assert read_table(table(found)) == found


def test_fixing_a_mixed_up_piece_moves_it_to_the_right_type():
    rows = table({"brick_2x4": 6})
    edit(rows, "Brick 2 x 4", 4)
    edit(rows, "Brick 1 x 4", 2)
    have = read_table(rows)
    assert have == {"brick_2x4": 4, "brick_1x4": 2}
    assert changes({"brick_2x4": 6}, have) == ["Brick 1 x 4: 0 → 2", "Brick 2 x 4: 6 → 4"]


def test_zero_or_empty_removes_a_piece():
    rows = table({"brick_2x2": 3, "brick_1x1": 1})
    edit(rows, "Brick 2 x 2", 0)
    edit(rows, "Brick 1 x 1", "")
    assert read_table(rows) == {}


@pytest.mark.parametrize("bad", ["three", -1, 2.5, 5000])
def test_bad_counts_give_a_readable_error(bad):
    rows = edit(table({"brick_2x2": 3}), "Brick 2 x 2", bad)
    with pytest.raises(ValueError, match="Brick 2 x 2|three"):
        read_table(rows)


def test_reads_a_dataframe_with_numbers_as_floats_and_text():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(table({"brick_2x2": 3}), columns=["Piece", "Model found", "You have", "Colors"])
    df["You have"] = df["You have"].astype(object)
    df.loc[df["Piece"] == "Brick 2 x 2", "You have"] = 4.0
    df.loc[df["Piece"] == "Plate 1 x 2", "You have"] = " 2 "
    df.loc[len(df)] = [None, None, None, None]  # a blank row someone added
    assert read_table(df) == {"brick_2x2": 4, "plate_1x2": 2}


def test_typed_in_names_and_ids_both_work():
    rows = [["brick 2 x 4", 0, 2, ""], ["slope45_2x2", 0, 1, ""], ["Brick  1 x 1", 0, 1, ""]]
    assert read_table(rows) == {"brick_2x4": 2, "slope45_2x2": 1, "brick_1x1": 1}


def test_edited_counts_change_the_ranking():
    # Start from exactly the parts of one build, then take a piece type away: its coverage must drop.
    from brickmatch.match import load_builds  # noqa: PLC0415
    build = load_builds()[0]
    exact = {p["piece"]: p["count"] for p in build["parts"]}
    assert next(r for r in rank_builds(read_table(table(exact)), min_coverage=0) if r["id"] == build["id"])["coverage"] == 1
    piece = build["parts"][0]["piece"]
    fewer = read_table(edit(table(exact), NAMES[piece], 0))
    after = next(r for r in rank_builds(fewer, min_coverage=0) if r["id"] == build["id"])
    assert after["coverage"] < 1 and after["missing"][piece] == build["parts"][0]["count"]
