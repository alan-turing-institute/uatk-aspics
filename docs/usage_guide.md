# Usage guide

## One-time installation

- [Poetry](https://python-poetry.org), for running a fork of the Python model
- The instructions assume you'e running in a shell on Linux or Mac, and have
  standard commands like `unzip` and `python3` available

```shell
git clone https://github.com/dabreegster/ua-aspics/
cd ua-aspics
# You only have to run this the first time, to install Python dependencies
poetry install
```

## Running the simulation

You first need to generate a synthetic population `.pb` file using [SPC](https://github.com/dabreegster/spc).

Convert the synthetic population file to a snapshot:

```shell
# Assuming the spc repository has been cloned in the parent directory of ua-aspics
# Note the naming scheme differs -- west_yorkshire_small becomes WestYorkshireSmall
poetry run python convert_snapshot.py -i ../spc/data/output/west_yorkshire_small.pb -o data/snapshots/WestYorkshireSmall/cache.npz
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
