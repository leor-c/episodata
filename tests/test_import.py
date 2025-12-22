import episodata

def test_imports():
    assert hasattr(episodata, "Episode")
    assert hasattr(episodata, "EpisodeDataset")
