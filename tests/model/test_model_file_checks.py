"""What ``learn-model`` says about a model path it cannot write.

Before anything is loaded or fitted, ``learn_model_wrapper`` checks that the
model file's extension is one ``save_dict`` can write (``.p``, ``.pkl``,
``.z``, ``.mat``, ``.h5``) and that its folder is writable. Both stop with an
error naming the path, what was found and what to do, and both hold under
``python -O``, which strips ``assert`` statements.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from moseq2_model.helpers.wrappers import learn_model_wrapper

ACCEPTED = ".p, .pkl, .z, .mat, .h5"


@pytest.mark.parametrize("name, found", [("model.txt", "'.txt'"), ("model", "no extension")])
def test_an_unknown_extension_names_the_path_and_the_accepted_ones(tmp_path, name, found):
    dest = tmp_path / name
    with pytest.raises(ValueError) as raised:
        learn_model_wrapper("unused_scores.h5", str(dest), {})
    message = str(raised.value)
    assert str(dest) in message
    assert found in message
    assert ACCEPTED in message
    assert "model.p" in message
    assert not dest.exists()


def test_the_extension_check_survives_python_O(tmp_path):
    dest = tmp_path / "model.txt"
    code = (
        "from moseq2_model.helpers.wrappers import learn_model_wrapper\n"
        f"learn_model_wrapper('unused_scores.h5', {str(dest)!r}, {{}})\n"
    )
    result = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
    assert "ValueError" in result.stderr and ACCEPTED in result.stderr


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root can write to a read-only folder")
@pytest.mark.skipif(os.name == "nt",
                    reason="on Windows the read-only attribute does not stop files being "
                           "created in a folder, so there is no unwritable folder to name")
def test_an_unwritable_folder_is_named(tmp_path):
    folder = tmp_path / "locked"
    folder.mkdir()
    folder.chmod(0o555)
    try:
        with pytest.raises(PermissionError) as raised:
            learn_model_wrapper("unused_scores.h5", str(folder / "model.p"), {})
    finally:
        folder.chmod(0o755)
    message = str(raised.value)
    assert str(folder) in message
    assert "permission" in message.lower()


def test_apply_model_checks_its_output_file_the_same_way(tmp_path):
    from moseq2_model.helpers.wrappers import apply_model_wrapper

    dest = tmp_path / "applied.txt"
    with pytest.raises(ValueError, match=r"has '\.txt'.*\.p, \.pkl, \.z, \.mat, \.h5"):
        apply_model_wrapper("unused_model.p", "unused_scores.h5", str(dest), {})
