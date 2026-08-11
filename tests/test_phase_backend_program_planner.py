from map.phase_backend_program_planner import (
    contiguous_block_mapping,
    congruence_mapping,
    demo_plan,
    kernel_class_count_from_frequency,
    plan_program,
)


def test_representative_character_frequency_has_requested_class_count():
    for n in [4, 6, 8, 12]:
        for m in range(2, n):
            if n % m == 0:
                f = n // m
                assert kernel_class_count_from_frequency(n, f) == m


def test_distinct_congruence_kernels_require_distinct_precarried_character_coordinates_in_restricted_strategy():
    n = 12
    p = plan_program(
        n,
        [
            ("C2", congruence_mapping(n, 2)),
            ("C3", congruence_mapping(n, 3)),
            ("C4", congruence_mapping(n, 4)),
        ],
    )
    assert p.precarried_carrier_count == 4  # faithful f=1 + one coordinate per quotient kernel
    assert p.exact_standing_character_lower_bound == 4
    assert p.precarried_character_bank == [1, 3, 4, 6]


def test_repeated_same_kernel_does_not_add_standing_carrier():
    n = 12
    q = congruence_mapping(n, 3)
    p = plan_program(n, [("a", q), ("b", q), ("c", q)])
    assert p.precarried_character_bank == [1, 4]
    assert p.exact_standing_character_lower_bound == 2


def test_contiguous_kernel_maps_to_shil_without_extra_character():
    n = 12
    p = plan_program(n, [("blocks", contiguous_block_mapping(n, 4))])
    assert p.precarried_character_bank == [1]
    assert p.shil_stage_count == 1
    assert p.transitions[0].lowering == "uniform SHIL basin collapse"


def test_unsupported_kernel_is_rejected_not_silently_optimized():
    n = 6
    p = plan_program(n, [("bad", [0, 0, 1, 0, 1, 1])])
    assert p.rejected_transition_count == 1
    assert p.transitions[0].lowering == "reject current phase library"
    assert p.transitions[0].reject_reason is not None


def test_demo_has_two_precarried_quotient_characters_one_shil_and_two_rejections():
    p = demo_plan()
    assert p.n == 12
    assert p.precarried_character_bank == [1, 4, 6]
    assert p.exact_standing_character_lower_bound == 3
    assert p.shil_stage_count == 1
    assert p.rejected_transition_count == 2
