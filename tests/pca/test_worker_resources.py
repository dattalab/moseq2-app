"""How much memory and how many CPUs PCA's local dask workers are sized from.

``get_env_cpu_and_mem`` sets the ceiling on the number of local workers and
the memory each is given. The machine's own numbers (``psutil``) describe the
whole computer, so in a container with a memory limit or on a shared server
they overstate what this process may use. The process's limits are its
cgroup's memory limit, the CPUs it may run on, and on SLURM the allocation
SLURM states in its environment variables. Each test fakes the machine, the
cgroup files and the environment, and checks the numbers that come back.
"""

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

import pytest

import moseq2_pca.util as util

GiB = 2**30
MiB = 2**20
#: The machine in every test: plenty of free memory and 96 CPUs.
MACHINE_AVAILABLE = 500 * GiB
MACHINE_CPUS = 96
#: What cgroup v1 reports when there is no limit.
V1_NO_LIMIT = 9223372036854771712

SLURM_VARIABLES = ("SLURM_JOBID", "SLURM_JOB_ID", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU",
                   "SLURM_CPUS_ON_NODE")


@pytest.fixture
def machine(monkeypatch, tmp_path):
    """A 96-CPU machine with 500 GiB free, outside SLURM, and empty cgroup files."""
    for name in SLURM_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(util.psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=MACHINE_AVAILABLE, total=MACHINE_AVAILABLE))
    monkeypatch.setattr(util.psutil, "cpu_count", lambda *a, **k: MACHINE_CPUS)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(MACHINE_CPUS)), raising=False)

    def no_sacct(*args, **kwargs):
        raise AssertionError(f"sacct was called: {args}")

    monkeypatch.setattr(subprocess, "check_output", no_sacct)
    proc = tmp_path / "proc_self_cgroup"
    root = tmp_path / "sys_fs_cgroup"
    root.mkdir()
    monkeypatch.setattr(util, "PROC_SELF_CGROUP", str(proc), raising=False)
    monkeypatch.setattr(util, "CGROUP_ROOT", str(root), raising=False)
    return SimpleNamespace(proc=proc, root=root)


def write_v2(machine, path, limit, current=0, inactive_file=0):
    """A cgroup v2 group at path with memory.max, memory.current and memory.stat."""
    machine.proc.write_text(f"0::{path}\n")
    folder = machine.root / path.lstrip("/")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "memory.max").write_text(f"{limit}\n")
    (folder / "memory.current").write_text(f"{current}\n")
    (folder / "memory.stat").write_text(f"anon 0\ninactive_file {inactive_file}\nactive_file 0\n")
    return folder


def write_v1(machine, path, limit, usage=0, inactive_file=0):
    """A cgroup v1 memory group at path."""
    machine.proc.write_text(f"12:memory:{path}\n11:cpu,cpuacct:{path}\n")
    folder = machine.root / "memory" / path.lstrip("/")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "memory.limit_in_bytes").write_text(f"{limit}\n")
    (folder / "memory.usage_in_bytes").write_text(f"{usage}\n")
    (folder / "memory.stat").write_text(f"cache 0\ntotal_inactive_file {inactive_file}\n")
    return folder


def test_no_limits_uses_the_machine(machine):
    """Without a cgroup limit or a CPU restriction, the machine's numbers are what there is."""
    mem, cpu = util.get_env_cpu_and_mem()
    assert mem == pytest.approx(MACHINE_AVAILABLE * 0.8)
    assert cpu == MACHINE_CPUS - 1


def test_cgroup_v2_limit_in_a_container(machine):
    """A container's 4 GiB limit, 1 GiB of it in use of which 0.5 GiB is reclaimable file cache."""
    machine.proc.write_text("0::/\n")
    for name, value in (("memory.max", 4 * GiB), ("memory.current", 1 * GiB)):
        (machine.root / name).write_text(f"{value}\n")
    (machine.root / "memory.stat").write_text(f"inactive_file {GiB // 2}\n")
    mem, _ = util.get_env_cpu_and_mem()
    assert mem == pytest.approx((4 * GiB - GiB // 2) * 0.8)


def test_cgroup_v2_limit_set_on_a_parent_group(machine):
    """The tightest limit along the path to the root binds, wherever it is set."""
    write_v2(machine, "/system.slice/job_1/step_0", "max")
    (machine.root / "system.slice" / "job_1" / "memory.max").write_text(f"{6 * GiB}\n")
    mem, _ = util.get_env_cpu_and_mem()
    assert mem == pytest.approx(6 * GiB * 0.8)


def test_cgroup_v2_no_limit_is_ignored(machine):
    write_v2(machine, "/user.slice", "max")
    mem, _ = util.get_env_cpu_and_mem()
    assert mem == pytest.approx(MACHINE_AVAILABLE * 0.8)


def test_cgroup_v1_limit(machine):
    write_v1(machine, "/docker/abc", 2 * GiB, usage=GiB, inactive_file=GiB // 4)
    mem, _ = util.get_env_cpu_and_mem()
    assert mem == pytest.approx((2 * GiB - GiB + GiB // 4) * 0.8)


def test_cgroup_v1_no_limit_is_ignored(machine):
    write_v1(machine, "/user.slice", V1_NO_LIMIT)
    mem, _ = util.get_env_cpu_and_mem()
    assert mem == pytest.approx(MACHINE_AVAILABLE * 0.8)


def test_cgroup_limit_above_what_the_machine_has_free(machine, monkeypatch):
    """A limit larger than the machine's free memory is capped by the machine."""
    write_v2(machine, "/", 64 * GiB)
    monkeypatch.setattr(util.psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=10 * GiB, total=64 * GiB))
    mem, _ = util.get_env_cpu_and_mem()
    assert mem == pytest.approx(10 * GiB * 0.8)


def test_cpus_the_process_may_run_on(machine, monkeypatch):
    """A process pinned to 4 of the machine's 96 CPUs sizes from the 4."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    _, cpu = util.get_env_cpu_and_mem()
    assert cpu == 3


def test_no_sched_getaffinity_uses_the_cpu_count(machine, monkeypatch):
    """macOS and Windows have no sched_getaffinity; the CPU count is used."""
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    _, cpu = util.get_env_cpu_and_mem()
    assert cpu == MACHINE_CPUS - 1


def test_slurm_memory_per_node(machine, monkeypatch):
    """SLURM_MEM_PER_NODE is in megabytes (MiB); SLURM_CPUS_ON_NODE is the allocation's CPUs."""
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOBID", "123")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "24576")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "4")
    mem, cpu = util.get_env_cpu_and_mem()
    assert mem == pytest.approx(24576 * MiB * 0.8)
    assert cpu == 3


def test_slurm_memory_per_cpu(machine, monkeypatch):
    """With --mem-per-cpu, the allocation is that times the CPUs on the node."""
    monkeypatch.setenv("SLURM_JOBID", "123")
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "4096")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
    mem, cpu = util.get_env_cpu_and_mem()
    assert mem == pytest.approx(8 * 4096 * MiB * 0.8)
    assert cpu == 7


@pytest.mark.parametrize("reqmem, expected", [
    ("2T", 2 * 2**40),
    ("4Gc", 4 * 4 * GiB),
    ("16Gn", 16 * GiB),
    ("16000M", 16000 * MiB),
], ids=["terabytes", "per-cpu", "per-node", "megabytes"])
def test_sacct_when_slurm_states_no_memory(machine, monkeypatch, reqmem, expected):
    """Without SLURM's memory variables, sacct's ReqMem is read, in any of its units."""
    monkeypatch.setenv("SLURM_JOBID", "123")
    calls = []

    def sacct(cmd, *args, **kwargs):
        calls.append(cmd)
        return f"4|{reqmem}|\n".encode()

    monkeypatch.setattr(subprocess, "check_output", sacct)
    monkeypatch.setattr(util.psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=100 * 2**40, total=100 * 2**40))
    mem, cpu = util.get_env_cpu_and_mem()
    assert calls
    assert mem == pytest.approx(expected * 0.8)
    assert cpu == 3
