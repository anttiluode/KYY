from map.phase_kernel_lowering_classifier import (
    character_kernel,
    classify_kernel,
    coverage_summary,
    shil_faithful_embeddings,
    verify_nontrivial_character_shil_disjoint,
)


def test_c4_adjacent_is_shil_not_character():
    c = classify_kernel([0, 0, 1, 1])
    assert c.character_candidates == []
    assert c.shil_faithful_embeddings
    assert c.primary_lowering.startswith("uniform SHIL")


def test_c4_alternating_is_quotient_character_not_shil():
    c = classify_kernel([0, 1, 0, 1])
    assert 2 in c.character_candidates
    assert c.shil_faithful_embeddings == []
    assert c.primary_lowering == "quotient-aligned non-faithful character"


def test_f2_character_has_alternating_c4_kernel():
    assert character_kernel(4, 2) == (0, 1, 0, 1)


def test_unequal_kernel_rejects_current_uniform_library():
    c = classify_kernel([0, 0, 1, 2])
    assert not c.equal_class_sizes
    assert c.character_candidates == []
    assert c.shil_faithful_embeddings == []
    assert c.reject_reason == "kernel classes have unequal sizes"


def test_equal_size_can_still_be_unsupported():
    c = classify_kernel([0, 0, 1, 0, 1, 1])
    assert c.equal_class_sizes
    assert c.character_candidates == []
    assert c.shil_faithful_embeddings == []
    assert "neither cyclic-contiguous nor cyclic-congruence" in c.reject_reason


def test_nontrivial_character_kernel_never_becomes_uniform_shil_under_faithful_reencoding_small_n():
    audit = verify_nontrivial_character_shil_disjoint(16)
    assert audit["collision_count"] == 0


def test_c4_exhaustive_partition_coverage():
    s = coverage_summary(4)
    assert s["bell_partition_count"] == 15
    assert s["nontrivial_partition_count"] == 13
    assert s["counts"] == {
        "trivial": 2,
        "character_only": 1,
        "shil_only": 2,
        "both_nontrivial": 0,
        "unsupported_equal_size": 0,
        "unsupported_unequal_size": 10,
    }


def test_c6_exhaustive_partition_coverage():
    s = coverage_summary(6)
    assert s["bell_partition_count"] == 203
    assert s["nontrivial_partition_count"] == 201
    assert s["counts"] == {
        "trivial": 2,
        "character_only": 2,
        "shil_only": 5,
        "both_nontrivial": 0,
        "unsupported_equal_size": 18,
        "unsupported_unequal_size": 176,
    }


def test_c8_exhaustive_partition_coverage():
    s = coverage_summary(8)
    assert s["bell_partition_count"] == 4140
    assert s["nontrivial_partition_count"] == 4138
    assert s["counts"] == {
        "trivial": 2,
        "character_only": 2,
        "shil_only": 12,
        "both_nontrivial": 0,
        "unsupported_equal_size": 126,
        "unsupported_unequal_size": 3998,
    }
