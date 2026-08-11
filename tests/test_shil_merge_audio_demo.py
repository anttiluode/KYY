import wave

from map.shil_merge_audio_demo import build_demo, simulate_source_trials


def test_fixed_seed_midpoint_is_ambiguous_and_compiled_is_stable():
    bad, _ = simulate_source_trials(1, 3.141592653589793 / 4.0, 12, 1234, 0.02)
    good, _ = simulate_source_trials(1, 3.141592653589793 / 8.0, 12, 1234, 0.02)
    assert 0 in set(int(x) for x in bad)
    assert 1 in set(int(x) for x in bad)
    assert set(int(x) for x in good) == {0}


def test_demo_writes_stereo_wav_and_metadata(tmp_path):
    wav_path = tmp_path / 'demo.wav'
    json_path = tmp_path / 'demo.json'
    payload = build_demo(wav_path, json_path, count=4, seed=1234)
    assert wav_path.exists()
    assert json_path.exists()
    assert payload['audio']['sample_rate'] == 48000
    with wave.open(str(wav_path), 'rb') as wf:
        assert wf.getnchannels() == 2
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 48000
        assert wf.getnframes() > 1000
