"""The SLURM options and the sbatch line, in one place.

``batch-extract`` and ``kappa-scan`` each declared the same six SLURM options
and built the same ``sbatch`` line in their own words. ``SlurmOptions`` holds
the six, and ``sbatch_command`` writes the line both commands write today.
"""

from __future__ import annotations

from moseq2_common.slurm import SlurmOptions, sbatch_command


def test_sbatch_command_matches_batch_extracts_line():
    assert sbatch_command("moseq2-extract extract --config-file c.yaml d.dat; ", ncpus=1,
                          memory="5GB", partition="short", wall_time="3:00:00", prefix="") == (
        'sbatch -c 1 --mem=5GB -p short -t 3:00:00 '
        '--wrap "moseq2-extract extract --config-file c.yaml d.dat; "'
    )


def test_sbatch_command_matches_kappa_scans_line():
    assert sbatch_command("moseq2-model learn-model s.h5 m.p --kappa 10", ncpus=8, memory="5GB",
                          partition="short", wall_time="3:00:00", prefix="module load x; ") == (
        'sbatch -c 8 --mem=5GB -p short -t 3:00:00 '
        '--wrap "module load x; moseq2-model learn-model s.h5 m.p --kappa 10"'
    )


def test_slurm_options_declare_the_shared_six():
    assert SlurmOptions().as_dict() == {
        "cluster_type": "local", "prefix": "", "memory": "5GB",
        "wall_time": "3:00:00", "partition": "short", "run_cmd": False,
    }
