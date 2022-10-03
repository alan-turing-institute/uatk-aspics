# Usage guide

WARNING: The simulation currently only runs in Manchester, and you must use
`config/new_parameters.yml`. All other config files are broken. You can revert
the git repository to September 2022 (commit
`03949aab8e72b384928439f6df9013073e11be96`) to restore behavior in other areas.

## One-time installation

- [Poetry](https://python-poetry.org), for running a fork of the Python model
- The instructions assume you'e running in a shell on Linux or Mac, and have
  standard commands like `unzip` and `python3` available

```shell
git clone https://github.com/alan-turing-institute/uatk-aspics/
cd ua-aspics
# You only have to run this the first time, to install Python dependencies
poetry install
```

## Running the simulation

You first need to download or generate a synthetic population `.pb` file using [SPC](https://github.com/alan-turing-institute/uatk-spc).

Convert the synthetic population file to a snapshot:

```shell
# Assuming the uatk-spc repository has been cloned in the parent directory of ua-aspics
# Note the naming scheme differs -- west_yorkshire_small becomes WestYorkshireSmall
poetry run python convert_snapshot.py -i ../uatk-spc/data/output/west_yorkshire.pb -o data/snapshots/WestYorkshireSmall/cache.npz
```

Then to run the snapshot file in the Python model:

```shell
poetry run python gui.py -p config/WestYorkshireSmall.yml
```

This should launch an interactive dashboard. Or you can run the simulation in
"headless" mode and instead write summary output data:

```shell
poetry run python headless.py -p config/WestYorkshireSmall.yml
```
