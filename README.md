# Episodata

Minimal episode + segment replay buffer for RL, built around **PyTorch** + **TensorDict**.

## Install

```bash
pip install -e .
```

## API

The package exports:

- `Episode`
- `EpisodeDataset`
- `SegmentsDataset`
- `Collector`

## Development

```bash
pip install -e .[dev]
pytest
```
