#!/usr/bin/env python3

import click
from aspics.loader import setup_sim
from aspics.inspector import Inspector


@click.command()
@click.option(
    "-p",
    "--parameters-file",
    type=click.Path(exists=True),
    help="Parameters file to use to configure the model. This must be located in the working directory.",
)
@click.option(
    "--spc",
    type=click.Path(exists=True),
    help="The same SPC protobuf file used with convert_snapshot.py.",
    default=None,
)
@click.option(
    "--events",
    type=click.Path(exists=True),
    help="Simulate people attending these large events. Path to a CSV file.",
    default=None,
)
def main(parameters_file, spc, events):
    simulator, snapshot, study_area, _ = setup_sim(parameters_file, spc, events)

    inspector = Inspector(
        simulator,
        snapshot,
        snapshot_folder=f"data/snapshots/{study_area}/",
        # Number of visualized connections per person
        nlines=4,
        window_name=study_area,
        width=2560,
        height=1440,
    )
    while inspector.is_active():
        inspector.update()


if __name__ == "__main__":
    main()
