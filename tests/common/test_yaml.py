"""The shared YAML helpers in moseq2-common.

Every YAML file the MoSeq2 packages read or write goes through
``moseq2_common.yaml``. Two things a user met before it existed:

- a numpy value (kappa from a model, a syllable id from ``np.unique``) made the
  safe writer raise ``RepresenterError`` (moseq2-viz#54);
- that failure came after the target was opened with ``"w"``, so the file being
  replaced, often the project's ``config.yaml``, was left empty.

numpy values are now written as the plain value they hold, and nothing is
written until the whole document has serialized. The format is unchanged: the
legacy ``safe_dump`` layout, read as YAML 1.2.
"""

from __future__ import annotations

import collections
import io
import os
import stat

import numpy as np
import pytest

from moseq2_common.yaml import UnwritableValueError, dump_yaml, read_yaml, write_yaml

NUMPY = {  # name: (numpy value, the plain value it must write as)
    "int64": (np.int64(212414), 212414),
    "int32": (np.int32(7), 7),
    "float64": (np.float64(1.5), 1.5),
    "float32": (np.float32(0.25), 0.25),
    "bool_": (np.bool_(True), True),
    "str_": (np.str_("rear"), "rear"),
    "array": (np.array([1, 2, 3]), [1, 2, 3]),
    "matrix": (np.array([[1.5, 2.0], [0.0, -1.0]]), [[1.5, 2.0], [0.0, -1.0]]),
    "zero_d": (np.array(4), 4),
}


@pytest.mark.parametrize("name", NUMPY)
def test_numpy_value_is_written_as_the_plain_value(name, tmp_path):
    value, plain = NUMPY[name]
    path = tmp_path / "out.yaml"
    write_yaml({"v": value, "nested": [{"v": value}], "tup": (value,)}, path)
    assert path.read_text() == dump_yaml({"v": plain, "nested": [{"v": plain}], "tup": [plain]})
    loaded = read_yaml(path)
    assert loaded["v"] == plain and type(loaded["v"]) is type(plain)


def test_kappa_from_a_model_is_written_as_an_integer(tmp_path):  # moseq2-viz#54
    write_yaml({"kappa": np.int64(212414)}, tmp_path / "info.yaml")
    assert (tmp_path / "info.yaml").read_text() == "{kappa: 212414}\n"


def test_numpy_keys_are_written_as_plain_keys():
    assert dump_yaml({np.int64(3): "rear"}) == "{3: rear}\n"


def test_float32_writes_its_exact_value():
    # .item() gives the float32's exact double; it is not rounded to 0.1.
    assert dump_yaml({"x": np.float32(0.1)}) == "{x: 0.10000000149011612}\n"


def test_the_callers_data_is_not_changed(tmp_path):
    value = np.int64(3)
    data = {"v": value, "l": [value]}
    write_yaml(data, tmp_path / "out.yaml")
    assert data["v"] is value and data["l"][0] is value


def test_unwritable_value_leaves_the_file_as_it_was(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_bytes(b"fps: 30\n")
    with pytest.raises(UnwritableValueError) as err:
        write_yaml({"a": {"b": [1, 1j]}}, path)
    assert path.read_bytes() == b"fps: 30\n"
    assert err.value.key_path == ("a", "b", 1)
    assert str(err.value) == (
        f"Cannot save {path}: the value at ['a']['b'][1] is 1j, of type complex, which a "
        f"YAML file cannot hold. {path} was left as it was. If you set this value, change "
        "it to text, a number, true/false, a list or a dictionary and run the step again; "
        "if MoSeq2 set it, please report this message at "
        "https://github.com/dattalab/moseq2-app/issues."
    )


def test_unwritable_key_is_named():
    with pytest.raises(UnwritableValueError) as err:
        dump_yaml({"a": {1j: 2}})
    assert "a key in the mapping at ['a'] is 1j, of type complex" in str(err.value)


def test_unwritable_value_writes_nothing_to_a_stream():
    stream = io.StringIO()
    stream.write("keep\n")
    with pytest.raises(UnwritableValueError):
        write_yaml({"a": 1, "b": 1j}, stream)
    assert stream.getvalue() == "keep\n"


@pytest.mark.parametrize(
    "value",
    [1j, collections.defaultdict(list), object()],
    ids=["complex", "defaultdict", "object"],
)
def test_other_types_the_safe_writer_refuses_still_raise(value):
    with pytest.raises(UnwritableValueError):
        dump_yaml({"v": value})


def test_a_failed_write_does_not_break_the_next(tmp_path):
    with pytest.raises(UnwritableValueError):
        write_yaml({"a": 1j}, tmp_path / "out.yaml")
    write_yaml({"a": 1}, tmp_path / "out.yaml")
    assert (tmp_path / "out.yaml").read_text() == "{a: 1}\n"


def test_a_symlinked_target_stays_a_symlink(tmp_path):
    real = tmp_path / "real.yaml"
    real.write_text("old: 1\n")
    link = tmp_path / "config.yaml"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"cannot create a symlink here: {exc}")
    write_yaml({"new": 2}, link)
    assert link.is_symlink() and real.read_text() == "{new: 2}\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_the_files_permissions_are_kept(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("a: 1\n")
    path.chmod(0o640)
    write_yaml({"a": 2}, path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_written_bytes_match_text_mode_writing(tmp_path):
    data = {"x": list(range(40)), "k": "é"}
    write_yaml(data, tmp_path / "a.yaml")
    with open(tmp_path / "b.yaml", "w") as handle:
        handle.write(dump_yaml(data))
    assert (tmp_path / "a.yaml").read_bytes() == (tmp_path / "b.yaml").read_bytes()


def test_output_keeps_the_legacy_format():
    # allow_unicode = False and width = 82, as safe_dump wrote them.
    assert dump_yaml({"k": "é"}) == '{k: "\\xE9"}\n'
    assert dump_yaml({"x": list(range(40))}) == (
        "x: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, \n"
        "    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]\n"
    )


def test_files_are_read_as_yaml_1_2():
    assert read_yaml(io.StringIO("wall_time: 06:00:00\nflag: yes\n")) == {
        "wall_time": "06:00:00",
        "flag": "yes",
    }
