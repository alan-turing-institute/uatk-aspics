# Usage guide

## One-time installation

- [Poetry](https://python-poetry.org), for running the Python model
- The instructions assume you'e running in a shell on Linux or Mac

```shell
git clone https://github.com/alan-turing-institute/uatk-aspics/
cd uatk-aspics
# You only have to run this the first time, to install Python dependencies
poetry install
```

## Running the simulation

You first need to download or generate a synthetic population `.pb` file using
[SPC](https://github.com/alan-turing-institute/uatk-spc).

Convert the synthetic population file to a snapshot:

```shell
# Assuming the uatk-spc repository has been cloned in the parent directory of ua-aspics
# Note the naming scheme differs -- west_yorkshire_small becomes WestYorkshireSmall
poetry run python convert_snapshot.py -i ../uatk-spc/data/output/west_yorkshire.pb -o data/snapshots/WestYorkshireSmall/cache.npz
```

Then to run the snapshot file in the Python model:

```shell
poetry run python gui.py -p config/WestYorkshireSmall.yml --spc ../uatk-spc/data/output/west_yorkshire.pb
```

This should launch an interactive dashboard. Or you can run the simulation in
"headless" mode and instead write summary output data:

```shell
poetry run python headless.py -p config/WestYorkshireSmall.yml --spc ../uatk-spc/data/output/west_yorkshire.pb
```

Note you provide the same `.pb` file that you used with `convert_snapshot.py`
when running the simulation. This is optional for most simulations, but
required to model large events.
