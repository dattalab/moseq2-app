"""A step's options are declared once, in a class, and resolved in one order.

Before ``moseq2_common.options`` each package kept its own merge of
``config.yaml`` with the command line (two ``command_with_config`` copies and
ten inline merges), and the merges disagreed: a null in ``config.yaml``
replaced a default on the notebook path but not on the command line, a scalar
for a ``multiple`` option reached frame cleaning as a scalar, and an option
typed at its default value lost to ``config.yaml`` (#424, moseq2-extract#24).

A class of ``option`` fields carries each setting's default, flags, type and
help. ``resolve`` layers class defaults, ``config.yaml``, the session's entry
and the values passed explicitly, in that order; ``click_options`` writes the
command's options from the class, in the order 1.5.0's help lists them; and
``explicit_values`` returns what the user typed, including a value typed at
its default.
"""

from __future__ import annotations

import dataclasses

import click
import pytest
from click.testing import CliRunner

from moseq2_common.options import (
    ConfigFileError,
    Options,
    arena_fields,
    click_options,
    explicit_values,
    option,
    read_config,
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Demo(Options):
    rank: int = option(25, "--rank", type=int, help="Number of PCs to keep")
    progressbar: bool = option(True, "--progressbar", type=bool, help="Show a progress bar")
    use_fft: bool = option(False, "--use-fft", is_flag=True, help="Use an FFT")
    nworkers: int = option(1, "-n", "--nworkers", type=int, help="Number of workers")
    gaussfilter_space: tuple = option(
        (1.5, 1), "--gaussfilter-space", type=(float, float), help="Spatial filter"
    )
    medfilter_space: tuple = option(
        (0,), "--medfilter-space", type=int, multiple=True, help="Median filter"
    )
    bg_roi_depth_range: tuple = option(
        (650, 750), "--bg-roi-depth-range", type=(float, float), arena=True,
        help="Range to search for floor of arena (in mm)",
    )


ORDER = ("rank", "input_dir", "progressbar", "use_fft", "nworkers",
         "gaussfilter_space", "medfilter_space", "bg_roi_depth_range")
INPUT_DIR = click.option("--input-dir", "-i", type=click.Path(), default="", help="Input folder")


def test_resolve_takes_defaults_then_config_then_entry_then_explicit():
    o = Demo.resolve(config={"rank": 30, "nworkers": 2, "use_fft": True},
                     entry={"rank": 40}, explicit={"nworkers": 3})
    assert (o.rank, o.nworkers, o.use_fft) == (40, 3, True)


def test_a_null_in_config_keeps_the_default():
    assert Demo.resolve(config={"rank": None}).rank == 25


def test_falsy_config_values_replace_defaults():
    o = Demo.resolve(config={"progressbar": False, "rank": 0, "medfilter_space": []})
    assert (o.progressbar, o.rank, o.medfilter_space) == (False, 0, ())


def test_a_scalar_or_list_becomes_a_tuple_for_a_tuple_field():
    o = Demo.resolve(config={"medfilter_space": 3, "gaussfilter_space": [2.25, 1.5]})
    assert (o.medfilter_space, o.gaussfilter_space) == ((3,), (2.25, 1.5))


def test_keys_the_class_does_not_declare_are_ignored():
    assert "camera_type" not in Demo.resolve(config={"camera_type": "azure"}).as_dict()


def test_an_undeclared_explicit_key_is_an_error():
    with pytest.raises(TypeError, match="bogus"):
        Demo.resolve(explicit={"bogus": 1})


def test_as_dict_holds_every_field_by_name():
    assert list(Demo().as_dict()) == [f.name for f in dataclasses.fields(Demo)]


def test_arena_fields_lists_the_fields_marked_arena():
    assert arena_fields(Demo) == ("bg_roi_depth_range",)


def hand_written():
    """The same options written as 1.5.0 writes them, in ORDER."""

    @click.group(context_settings={"show_default": True})
    def group():
        pass

    @group.command()
    @click.option("--rank", type=int, default=25, help="Number of PCs to keep")
    @INPUT_DIR
    @click.option("--progressbar", type=bool, default=True, help="Show a progress bar")
    @click.option("--use-fft", is_flag=True, default=False, help="Use an FFT")
    @click.option("-n", "--nworkers", type=int, default=1, help="Number of workers")
    @click.option("--gaussfilter-space", type=(float, float), default=(1.5, 1), help="Spatial filter")
    @click.option("--medfilter-space", type=int, multiple=True, default=(0,), help="Median filter")
    @click.option("--bg-roi-depth-range", type=(float, float), default=(650, 750),
                  help="Range to search for floor of arena (in mm)")
    def run(**kwargs):
        pass

    return group


def generated(callback=None):
    @click.group(context_settings={"show_default": True})
    def group():
        pass

    @group.command()
    @click_options(Demo, order=ORDER, others={"input_dir": INPUT_DIR})
    @click.pass_context
    def run(ctx, **kwargs):
        if callback is not None:
            callback(ctx, kwargs)

    return group


def described(param):
    return (param.name, param.opts, param.type.to_info_dict(), param.default, param.multiple,
            getattr(param, "is_flag", False), param.help)


def test_generated_help_equals_hand_written_help():
    runner = CliRunner()
    want = runner.invoke(hand_written(), ["run", "--help"])
    got = runner.invoke(generated(), ["run", "--help"])
    assert want.exit_code == got.exit_code == 0
    assert got.output == want.output
    assert ([described(p) for p in generated().commands["run"].params]
            == [described(p) for p in hand_written().commands["run"].params])


def test_order_must_name_every_option_once():
    without_nworkers = tuple(name for name in ORDER if name != "nworkers")
    with pytest.raises(ValueError, match="nworkers"):
        click_options(Demo, order=without_nworkers, others={"input_dir": INPUT_DIR})
    with pytest.raises(ValueError, match="rank"):
        click_options(Demo, order=ORDER + ("rank",), others={"input_dir": INPUT_DIR})
    with pytest.raises(ValueError, match="bogus"):
        click_options(Demo, order=ORDER + ("bogus",), others={"input_dir": INPUT_DIR})


def invoke(args):
    seen = {}

    def record(ctx, kwargs):
        seen["kwargs"] = kwargs
        seen["explicit"] = explicit_values(ctx, Demo)

    result = CliRunner().invoke(generated(record), ["run", *args])
    assert result.exit_code == 0, result.output
    return seen


def test_an_untyped_flag_is_false():
    seen = invoke([])
    assert seen["kwargs"]["use_fft"] is False
    assert Demo.resolve(explicit=seen["explicit"]).as_dict()["use_fft"] is False


def test_explicit_values_sees_an_option_typed_at_its_default():
    assert invoke(["--rank", "25"])["explicit"] == {"rank": 25}


@pytest.mark.parametrize(
    "content", [None, "", "a: [", "- 1\n- 2\n"],
    ids=["missing", "empty", "unparsable", "not-a-mapping"],
)
def test_read_config_names_a_missing_empty_or_unreadable_file(tmp_path, content):
    path = tmp_path / "config.yaml"
    if content is not None:
        path.write_text(content)
    with pytest.raises(ConfigFileError) as caught:
        read_config(path)
    assert str(path) in str(caught.value)
