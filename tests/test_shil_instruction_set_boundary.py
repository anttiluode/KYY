import importlib.util
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NAME="shil_instruction_set_boundary_for_tests"
SPEC=importlib.util.spec_from_file_location(NAME,ROOT/"map"/"shil_instruction_set_boundary.py")
assert SPEC and SPEC.loader
m=importlib.util.module_from_spec(SPEC);sys.modules[NAME]=m;SPEC.loader.exec_module(m)


def test_adjacent_pair_quotient_is_reachable_but_alternating_is_not():
    rec=m.reachable_partitions(4)
    adjacent=m.canonicalize_labels((0,0,1,1))
    alternating=m.canonicalize_labels((0,1,0,1))
    assert adjacent in rec
    assert alternating not in rec


def test_all_one_circle_reachable_partitions_have_contiguous_cyclic_fibers():
    rec=m.reachable_partitions(8)
    assert all(m.contiguous_cyclic_fibers(partition) for partition in rec)


def test_second_harmonic_realizes_c4_alternating_partition():
    h2,meta=m.harmonic_partition(4,2)
    assert h2 == m.canonicalize_labels((0,1,0,1))
    assert meta["output_phase_count"] == 2


def test_general_harmonic_kernel_is_congruence_class_partition():
    p,meta=m.harmonic_partition(12,3)
    assert meta["output_phase_count"] == 4
    # q and q+4 share the same third-harmonic phase.
    assert p[0] == p[4] == p[8]
    assert p[0] != p[1]
