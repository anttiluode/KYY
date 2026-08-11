import importlib.util
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NAME="shil_cyclic_quotient_for_tests"
SPEC=importlib.util.spec_from_file_location(NAME,ROOT/"map"/"shil_cyclic_quotient_compiler.py")
assert SPEC and SPEC.loader
m=importlib.util.module_from_spec(SPEC);sys.modules[NAME]=m;SPEC.loader.exec_module(m)


def test_c4_pair_merge_recovers_pi_over_8_plan():
    p=m.compile_equal_block_quotient(4,2)
    assert p.block_size == 2
    assert p.parity == "even"
    assert abs(p.blocks[0].coarse_attractor_phase-math.pi/8) < 1e-12
    assert abs(p.certified_margin-math.pi/8) < 1e-12


def test_odd_block_quotient_uses_middle_fine_state_and_has_half_spacing_margin():
    p=m.compile_equal_block_quotient(12,4)  # r=3, Delta=pi/6
    assert p.block_size == 3
    assert p.parity == "odd"
    assert p.blocks[0].representative_state == 1
    assert abs(p.blocks[0].coarse_attractor_phase-math.pi/6) < 1e-12
    assert abs(p.certified_margin-math.pi/12) < 1e-12


def test_even_block_margin_is_quarter_of_fine_spacing_independent_of_block_size():
    for n,mv in [(12,3),(16,4),(100,10)]:
        p=m.compile_equal_block_quotient(n,mv)
        assert p.block_size % 2 == 0
        assert abs(p.certified_margin-p.fine_spacing/4) < 1e-12


def test_uniform_shil_rejects_interleaved_kernel_classes():
    good=m.single_uniform_shil_realizable([0,0,1,1])
    bad=m.single_uniform_shil_realizable([0,1,0,1])
    assert good["realizable"]
    assert not bad["realizable"]


def test_uniform_shil_rejects_unequal_cyclic_kernel_classes():
    assert not m.single_uniform_shil_realizable([0,0,0,1,1,2])["realizable"]
