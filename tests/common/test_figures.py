"""Every figure a step shows is also saved: a PNG, a PDF and the run's record, together."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from moseq2_common.figures import save_figure  # noqa: E402
from moseq2_common.records import read_record, start_record  # noqa: E402


def test_save_figure_writes_png_pdf_and_record_together(tmp_path):
    figure, axis = plt.subplots()
    axis.plot([0, 1, 2], [2, 1, 3])
    record = start_record("train-pca", {"npcs": 3}).finish({"files": ["_pca/pca_scree.png"]})

    written = save_figure(figure, tmp_path / "_pca" / "pca_scree", record)
    plt.close(figure)

    assert written == [tmp_path / "_pca" / "pca_scree.png", tmp_path / "_pca" / "pca_scree.pdf"]
    assert written[0].read_bytes().startswith(b"\x89PNG")
    assert written[1].read_bytes().startswith(b"%PDF")
    assert read_record(written[0])["step"] == "train-pca"
    assert sorted(p.name for p in (tmp_path / "_pca").iterdir()) == ["pca_scree.pdf", "pca_scree.png",
                                                                     "pca_scree.yaml"]
