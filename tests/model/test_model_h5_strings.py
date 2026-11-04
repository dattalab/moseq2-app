"""A model file saved as ``.h5`` reads its session uuids back as ``str``.

``save_dict`` writes a list of strings, such as a model's ``keys`` and its
``metadata`` uuids, as variable-length UTF-8 text, and ``load_dict`` reads it
back through ``moseq2_common.h5``, the reader every other package uses. That
reader decodes such text to ``str``, so the uuids match the ones in the index
and in the PCA scores file. A single string value (a path, a group name) is
stored as ASCII and reads back as ``bytes``, as the shared reader does for
every package.
"""

from __future__ import annotations

from moseq2_model.util import load_dict, save_dict

UUIDS = ["5c72bf30-9596-4d4d-ae38-db9a7a28e912", "0b1f2a44-6c1d-4e0a-9d3b-2f6e8a7c1d55"]


def test_a_model_saved_as_h5_reads_its_uuids_as_str(tmp_path):
    model_file = str(tmp_path / "model.h5")
    save_dict(model_file, {
        "keys": UUIDS,
        "metadata": {"uuids": UUIDS, "groups": {UUIDS[0]: "saline", UUIDS[1]: "amphetamine"}},
        "pc_score_path": "/data/proj/_pca/pca_scores.h5",
    })

    model = load_dict(model_file)

    assert model["keys"].tolist() == UUIDS
    assert all(type(k) is str for k in model["keys"])
    assert model["metadata"]["uuids"].tolist() == UUIDS
    assert all(type(u) is str for u in model["metadata"]["uuids"])
    assert set(model["metadata"]["groups"]) == set(UUIDS)
    assert model["metadata"]["groups"][UUIDS[0]] == b"saline"
    assert model["pc_score_path"] == b"/data/proj/_pca/pca_scores.h5"
