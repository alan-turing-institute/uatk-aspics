# Generic functions that are used in the experiments notebooks
# Useful to put them in here so that they can be shared across notebooks
# and can be tested (see tests/experiements/opencl_runner_tests.py)
import os
from yaml import load, SafeLoader
import numpy as np
import multiprocessing
import itertools
import time
import tqdm
import pandas as pd
import random
import sys, os
sys.path.append('../')
import headless
from typing import List
from aspics.simulator import Simulator
from aspics.snapshot import Snapshot
from aspics.params import Params, IndividualHazardMultipliers, LocationHazardMultipliers
from aspics.disease_statuses import DiseaseStatus
from aspics.summary import Summary


class OpenCLRunner:
    """
    Includes useful functions for running the OpenCL model in notebooks
    This has been adapted to run in the ASPICS model.
    
    """

    # Need a list to optionally store additional constant parameter values that cannot
    # be passed through one of the run model functions.
    constants = {}

    @classmethod
    def setup_sim(cs,
                  parameters_file):

        print(f"Running a simulation based on {parameters_file}")
        
        try:
            with open(parameters_file, "r") as f:
                parameters = load(f, Loader=SafeLoader)
                sim_params = parameters["microsim"]
                calibration_params = parameters["microsim_calibration"]
                disease_params = parameters["disease"]
                iterations = sim_params["iterations"]
                study_area = sim_params["study-area"]
                output = sim_params["output"]
                output_every_iteration = sim_params["output-every-iteration"]
                use_lockdown = sim_params["use-lockdown"]
                start_date = sim_params["start-date"]
        except Exception as error:
            print("Error in parameters file format")
            raise error

        # Check the parameters are sensible
        if iterations < 1:
            raise ValueError("Iterations must be > 1")
        if (not output) and output_every_iteration:
            raise ValueError(
                "Can't choose to not output any data (output=False) but also write the data at every "
                "iteration (output_every_iteration=True)"
            )

        # Load the snapshot file
        snapshot_path = f"data/snapshots/{study_area}/cache.npz"
        if not os.path.exists(snapshot_path):
            raise Exception(
                f"Missing snapshot cache {snapshot_path}. Run SPC and convert_snapshot.py first to generate it."
            )
        print(f"Loading snapshot from {snapshot_path}")
        snapshot = Snapshot.load_full_snapshot(path=snapshot_path)
        print(f"Snapshot is {int(snapshot.num_bytes() / 1000000)} MB")

        # Apply lockdown values
        if use_lockdown:
            # Skip past the first entries
            snapshot.lockdown_multipliers = snapshot.lockdown_multipliers[start_date:]
        else:
            # No lockdown
            snapshot.lockdown_multipliers = np.ones(iterations + 1)

        # set the random seed of the model
        snapshot.seed_prngs(42)

        # set params
        if calibration_params is not None and disease_params is not None:
            snapshot.update_params(create_params(calibration_params, disease_params))

            if disease_params["improve_health"]:
                print("Switching to healthier population")
                snapshot.switch_to_healthier_population()

        # Create a simulator and upload the snapshot data to the OpenCL device
        simulator = Simulator(snapshot, parameters_file, gpu=True)
        [people_statuses, people_transition_times] = simulator.seeding_base()
        simulator.upload_all(snapshot.buffers)
        simulator.upload("people_statuses", people_statuses)
        simulator.upload("people_transition_times", people_transition_times)

        return simulator, snapshot, study_area, iterations
    
    @classmethod
    def update(cls, iterations: int = None, repetitions: int = None, observations: pd.DataFrame = None,
               use_gpu: bool = None, use_healthier_pop=None, store_detailed_counts: bool = None,
               parameters_file: str = None,
               opencl_dir: str = None, snapshot_filepath: str = None):
        """
        Update any of the variables that have already been initialised
        """
        if not cls.initialised:
            raise Exception("The OpenCLRunner class needs to be initialised first; call OpenCLRunner.init()")
        if iterations is not None:
            cls.ITERATIONS = iterations
        if repetitions is not None:
            cls.REPETITIONS = repetitions
        if observations is not None:
            cls.OBSERVATIONS = observations
        if use_gpu is not None:
            cls.USE_GPU = use_gpu
        if use_healthier_pop is not None:
            cls.USE_HEALTHIER_POP = use_healthier_pop
        if store_detailed_counts is not None:
            cls.STORE_DETAILED_COUNTS = store_detailed_counts
        if parameters_file is not None:
            cls.PARAMETERS_FILE = parameters_file
        if opencl_dir is not None:
            cls.OPENCL_DIR = opencl_dir
        if snapshot_filepath is not None:
            cls.SNAPSHOT_FILEPATH = snapshot_filepath



