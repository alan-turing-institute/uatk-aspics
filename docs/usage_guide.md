# Usage guide

Note these instructions currently fail for Mac M1 due to a mix of OpenCL,
pandas, and OpenGL issues. We're working on it, and will update this page once
resolved.

## One-time installation

- [Poetry](https://python-poetry.org), for running a fork of the Python model
  - If you have trouble installing Python dependencies -- especially on Mac M1
    -- you can instead use
    [conda](https://docs.conda.io/projects/conda/en/latest/index.html)
- The instructions assume you'e running in a shell on Linux or Mac, and have
  standard commands like `unzip` and `python3` available

```shell
git clone https://github.com/dabreegster/ua-aspics/
cd ua-aspics
# You only have to run this the first time, to install Python dependencies
poetry install
```

## Running the simulation

You first need to generate a synthetic population .pb file using [SPC](https://github.com/dabreegster/spc). Copy that into `data/processed_data/STUDY_AREA_NAME/synthpop.pb`. Note the naming scheme differs between the two projects -- `data/output/west_yorkshire_small.pb` from SPC should become `data/processed_data/WestYorkshireSmall/synthpop.pb`.

Convert the synthetic population file to a snapshot:

```shell
poetry run python convert_snapshot.py -i data/processed_data/WestYorkshireSmall/synthpop.pb -o data/processed_data/WestYorkshireSmall/snapshot/cache.npz
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

## Conda alternative

If `poetry` doesn't work, we also have a Conda environment. You can use it like
this:

```shell
conda env create -f environment.yml
conda activate aspics
python3.7 gui.py -p config/WestYorkshireSmall.yml
```

Note inside the Conda environment, just `python` may not work; specify
`python3.7`.

If you get
`CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.`
and the provided instructions don't help, on Linux you can try doing
`source ~/anaconda3/etc/profile.d/conda.sh`.
