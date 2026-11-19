# Contributing to GNSS-Sentinel

Thanks for your interest. GNSS-Sentinel is a passive, defensive indicator toolkit validated on synthetic data. Contributions must keep it that way: no transmission or RF-generation code.

## Getting started

Python 3 and numpy.

```
python -m gnss_sentinel gen meacon -o m.csv --seed 11
python -m gnss_sentinel stream m.csv
```

## Testing

```
python -m pytest -q
```

`python -m gnss_sentinel evaluate --quick --out results` regenerates results (the full run takes about 15 minutes). Label any new result as synthetic unless it uses real receiver data.

## Pull requests

- Open an issue first for larger changes.
- Keep PRs focused and fill in the pull request template.
- Follow `.editorconfig` and match the style of surrounding code.

## Conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
