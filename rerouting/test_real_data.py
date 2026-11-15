"""smoke test of the real-data pipeline on a small fake dataset in the geant format (not real results)."""
import real_data
import make_fake_geant


def test_pipeline_runs_on_fake_geant_format(tmp_path):
    make_fake_geant.make(str(tmp_path), steps=300)
    res = real_data.main(str(tmp_path / "tm"), str(tmp_path / "topology.csv"), out=str(tmp_path / "out"))
    assert set(res) == {"optimal", "never-switch", "greedy", "threshold", "overlap-aware"}
    assert all(c > 0 for c, _ in res.values())
