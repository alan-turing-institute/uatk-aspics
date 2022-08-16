# Code based on the https://github.com/Urban-Analytics/RAMP-UA/blob/2e65fea765f90549478049847115dde8bf223f6d/experiments/calibration/opencl_runner.py
# Adapted by Fernando Benitez-Paez

import os
import numpy as np
import multiprocessing
import itertools
import yaml
import time
import tqdm
import pickle
import pandas as pd
import random
import sys

from yaml import load, SafeLoader

sys.path.append('../')
from headless import run_headless
from typing import List
from aspics.simulator import Simulator  # Not an elegant way to address path issues.
from aspics.snapshot import Snapshot
from aspics.loader import create_params
from aspics.params import Params, IndividualHazardMultipliers, LocationHazardMultipliers
from aspics.disease_statuses import DiseaseStatus
from aspics.summary import Summary


class OpenCLRunner:
    constants = {}

    @classmethod
    def init(cls,
             repetitions: int,
             iterations: int,
             study_area: str,
             output: bool,
             output_every_iteration: bool,
             use_lockdown: bool,
             start_date: int,
             observations: pd.DataFrame,
             use_gpu: bool,
             use_healthier_pop: bool,
             store_detailed_counts: bool,
             parameters_file: str,
             ):

        cls.REPETITIONS = repetitions
        cls.ITERATIONS = iterations
        cls.STUDY_AREA = study_area
        cls.OUTPUT = output
        cls.OUTPUT_EVERY_ITERATION = output_every_iteration
        cls.USE_LOCKDOWN = use_lockdown
        cls.START_DATE = start_date
        cls.OBSERVATIONS = observations
        cls.USE_GPU = use_gpu
        cls.USE_HEALTHIER_POP = use_healthier_pop
        cls.STORE_DETAILED_COUNTS = store_detailed_counts
        cls.PARAMETERS_FILE = parameters_file
        cls.initialised = True

    @classmethod
    def update(cls,
               repetitions: int = None,
               iterations: int = None,
               study_area: str = None,
               output: bool = None,
               output_every_iteration: bool = None,
               use_lockdown: bool = None,
               start_date: int = None,
               observations: pd.DataFrame = None,
               use_gpu: bool = None,
               use_healthier_pop=None,
               store_detailed_counts: bool = None,
               parameters_file: str = None,
               ):
        """
        Update any of the variables that have already been initialised
        """
        if not cls.initialised:
            raise Exception("The OpenCLRunner class needs to be initialised first; call OpenCLRunner.init()")
        if iterations is not None:
            cls.ITERATIONS = iterations
        if repetitions is not None:
            cls.REPETITIONS = repetitions
        if study_area is not None:
            cls.STUDY_AREA = study_area
        if output is not None:
            cls.OUTPUT = output
        if output_every_iteration is not None:
            cls.OUTPUT_EVERY_ITERATION = output_every_iteration
        if use_lockdown is not None:
            cls.USE_LOCKDOWN = use_lockdown
        if start_date is not None:
            cls.START_DATE = start_date
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

    @classmethod  ### To ask Nick.
    def set_constants(cls, constants):
        """Set any constant variables (parameters) that override the defaults.
        :param constants: This should be a dist of parameter_nam -> value
        """
        cls.constants = constants

    @classmethod  ### Ask Nick about the use of this.
    def clear_constants(cls):
        cls.constants = {}

    @staticmethod
    def fit_l2(obs: np.ndarray, sim: np.ndarray):

        """Calculate the fitness of a model.

         Parameters
        ----------
        obs : array_like
              The observations data..
        sim : array_like
              The simulated data."""

        if len(obs) != len(sim):
            raise Exception(f"Lengths should be the same, not {len(obs)}) and {len(sim)}")
        if np.array(obs).shape != np.array(sim).shape:
            raise Exception("fShapes should be the same")

        return np.linalg.norm(np.array(obs) - np.array(sim))

    @staticmethod
    def get_mean_total_counts(summaries, disease_status: int, get_sd=False):
        """
        Get the mean total counts for a given disease status at every iteration over a number of model repetitions

        :param summaries: A list of Summary objects created by running the OpenCL model
        :param disease_status: The disease status number, e.g.  `DiseaseStatus.Exposed.value`
        :param get_sd: Optionally get the standard deviation as well

        :return: The mean total counts of the disease status per iteration, or (if get_sd=True)
            or a tuple of (mean,sd)

        """
        reps = len(summaries)  # Number of repetitions
        iters = len(summaries[0].total_counts[disease_status])  # Number of iterations for each repetition
        matrix = np.zeros(shape=(reps, iters))
        for rep in range(reps):
            matrix[rep] = summaries[rep].total_counts[disease_status]
        mean = np.mean(matrix, axis=0)
        sd = np.std(matrix, axis=0)

        if get_sd:
            return mean, sd
        else:
            return mean

    @staticmethod
    def get_cumulative_new_infections(summaries):
        """
        Get cumulative infections per day by summing all the non-susceptible people

        :param summaries: A list of Summary objects created by running the OpenCL model
        """
        iters = len(summaries[0].total_counts[DiseaseStatus.Exposed.value])  # Number of iterations for each repetition
        total_not_susceptible = np.zeros(iters)  # Total people not susceptible per iteration
        for d, disease_status in enumerate(DiseaseStatus):
            if disease_status != DiseaseStatus.Susceptible:
                mean = OpenCLRunner.get_mean_total_counts(summaries, d)  # Mean number of people with that disease
                total_not_susceptible = total_not_susceptible + mean
        return total_not_susceptible

    @staticmethod
    def create_params_manually(
            # Microsim params
            parameters_file: str = None,
            study_area: str = None,
            # dissease Params
            current_risk_beta: float = None,
            infection_log_scale: float = None,
            infection_mode: float = None,
            # hazard_individual_multipliers_params
            presymptomatic: float = None,
            asymptomatic: float = None,
            symptomatic: float = None,
            # hazard_location_multipliers_params
            retail: float = None,
            primary_school: float = None,
            secondary_school: float = None,
            home: float = None,
            work: float = None,
    ):

        print(f"Creating parameters manually based on values from Notebook")

        # Read the default parameters from a yml file, then override with any provided
        if parameters_file is None:
            parameters_file = f"config/{study_area}.yml"
            print(f"I read the parameter file '{study_area}")
        elif not os.path.isfile(parameters_file):
            raise Exception(f"The given parameters file '{parameters_file} is not a file, check the /config folder")

        with open(parameters_file) as f:
            parameters = yaml.load(f, Loader=yaml.SafeLoader)

        # sim_params = parameters["microsim"]  # Parameters for the dynamic microsim (python)
        calibration_params = parameters["microsim_calibration"]
        disease_params = parameters["disease"]  # Parameters for the disease model (r)

        # current_risk_beta needs to be set first  as the OpenCL model pre-multiplies the hazard multipliers by it
        current_risk_beta = OpenCLRunner._check_if_none("current_risk_beta",
                                                        current_risk_beta,
                                                        disease_params['current_risk_beta'])

        # Location hazard multipliers can be passed straight through to the LocationHazardMultipliers object.
        # If no argument was passed then the default in the parameters file is used. Note that they need to
        # be multiplied by the current_risk_beta
        location_hazard_multipliers = LocationHazardMultipliers(
            retail=current_risk_beta * OpenCLRunner._check_if_none("retail",
                                                                   retail,
                                                                   calibration_params["hazard_location_multipliers"][
                                                                       "Retail"]),
            primary_school=current_risk_beta * OpenCLRunner._check_if_none("primary_school",
                                                                           primary_school, calibration_params[
                                                                               "hazard_location_multipliers"][
                                                                               "PrimarySchool"]),
            secondary_school=current_risk_beta * OpenCLRunner._check_if_none("secondary_school",
                                                                             secondary_school, calibration_params[
                                                                                 "hazard_location_multipliers"][
                                                                                 "SecondarySchool"]),
            home=current_risk_beta * OpenCLRunner._check_if_none("home",
                                                                 home,
                                                                 calibration_params["hazard_location_multipliers"][
                                                                     "Home"]),
            work=current_risk_beta * OpenCLRunner._check_if_none("work",
                                                                 work,
                                                                 calibration_params["hazard_location_multipliers"][
                                                                     "Work"]),
        )

        # Individual hazard multipliers can be passed straight through
        individual_hazard_multipliers = IndividualHazardMultipliers(
            presymptomatic=OpenCLRunner._check_if_none("presymptomatic",
                                                       presymptomatic,
                                                       calibration_params["hazard_individual_multipliers"][
                                                           "presymptomatic"]),
            asymptomatic=OpenCLRunner._check_if_none("asymptomatic",
                                                     asymptomatic, calibration_params["hazard_individual_multipliers"][
                                                         "asymptomatic"]),
            symptomatic=OpenCLRunner._check_if_none("symptomatic",
                                                    symptomatic,
                                                    calibration_params["hazard_individual_multipliers"]["symptomatic"])
        )

        # Some parameters are set in the default.yml file and can be overridden
        pass  # None here yet

        obesity_multipliers = np.array(
            [disease_params["overweight"],
             disease_params["obesity_30"],
             disease_params["obesity_35"],
             disease_params["obesity_40"]])

        cvd = disease_params["cvd"]
        diabetes = disease_params["diabetes"]
        bloodpressure = disease_params["bloodpressure"]
        overweight_sympt_mplier = disease_params["overweight_sympt_mplier"]

        p = Params(
            location_hazard_multipliers=location_hazard_multipliers,
            individual_hazard_multipliers=individual_hazard_multipliers,
        )

        # Remaining parameters are defined within the Params class and have to be manually overridden
        if infection_log_scale is not None:
            p.infection_log_scale = infection_log_scale
        if infection_mode is not None:
            p.infection_mode = infection_mode

        p.obesity_multipliers = obesity_multipliers
        p.cvd_multiplier = cvd
        p.diabetes_multiplier = diabetes
        p.bloodpressure_multiplier = bloodpressure
        p.overweight_sympt_mplier = overweight_sympt_mplier

        return p

    @classmethod
    def _check_if_none(cls,
                       param_name,
                       param_value,
                       default_value):
        """Checks whether the given param is None. If so, it will return a constant value, if it has
         one, or failing that the provided default if it is"""
        if param_value is not None:
            # The value has been provided. Return it, but check a constant hasn't been set as well
            # (it's unlikely that someone would set a constant and then also provide a value for the same parameter)
            if param_name in cls.constants.keys():
                raise Exception(f"A parameter {param_name} has been provided, but it has also been set as a constant")
            return param_value
        else:  # No value provided, return a constant, if there is one, or the default otherwise
            if param_name in cls.constants.keys():
                return cls.constants[param_name]
            else:
                return default_value

    @staticmethod
    def run_aspics(i: int,
                   iterations: int,
                   study_area: str,
                   params,
                   output: bool,
                   output_every_iteration: bool,
                   use_gpu: bool,
                   use_lockdown: bool,
                   start_date: int,
                   calibration_params,
                   disease_params,
                   use_healthier_pop: bool,
                   store_detailed_counts: bool = True,
                   quiet=False) -> (np.ndarray, np.ndarray):

        print(f"Running a simulation ", {i}, " based on the study area")
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

        prev_obesity = np.copy(snapshot.buffers.people_obesity)
        if use_healthier_pop:
            snapshot.switch_to_healthier_population()

        print("testing obesity arrays not equal")
        print(np.mean(prev_obesity))
        print(np.mean(snapshot.buffers.people_obesity))

        # Apply lockdown values
        if use_lockdown:
            # Skip past the first entries
            snapshot.lockdown_multipliers = snapshot.lockdown_multipliers[start_date:]
        else:
            # No lockdown
            snapshot.lockdown_multipliers = np.ones(iterations + 1)

        # set the random seed of the model
        snapshot.seed_prngs(i)

        # set params
        if calibration_params is not None and disease_params is not None:
            snapshot.update_params(params)

            if disease_params["improve_health"]:
                print("Switching to healthier population")
                snapshot.switch_to_healthier_population()

        # Create a simulator and upload the snapshot data to the OpenCL device
        simulator = Simulator(snapshot, study_area, gpu=True)
        [people_statuses, people_transition_times] = simulator.seeding_base()
        simulator.upload_all(snapshot.buffers)
        simulator.upload("people_statuses", people_statuses)
        simulator.upload("people_transition_times", people_transition_times)

        if not quiet:
            print(f"Running simulation {i + 1}.")

        summary, final_state = run_headless(simulator, snapshot, iterations, quiet=True,
                                            store_detailed_counts=store_detailed_counts)
        return summary, final_state

    @staticmethod
    def run_aspics_multi(
            repetitions: int,
            iterations: int,
            study_area: str,
            params,
            output: bool,
            output_every_iteration: bool,
            use_gpu: bool,
            use_healthier_pop: bool,
            use_lockdown: bool,
            start_date: int,
            calibration_params,
            disease_params,
            store_detailed_counts: bool = False,
            multiprocess=False,
            random_ids=False
    ):

        # Prepare the function arguments. We need one set of arguments per repetition
        l_i = [i for i in range(repetitions)] if not random_ids else \
            [random.randint(1, 100000) for _ in range(repetitions)]
        l_iterations = [iterations] * repetitions
        l_study_area = [study_area] * repetitions
        l_params = [params] * repetitions
        l_output = [output] * repetitions
        l_output_every_iteration = [output_every_iteration] * repetitions
        l_use_gpu = [use_gpu] * repetitions
        l_use_lockdown = [use_lockdown] * repetitions
        l_start_date = [start_date] * repetitions
        l_calibration_params = [calibration_params] * repetitions
        l_disease_params = [disease_params] * repetitions
        l_use_healthier_pop = [use_healthier_pop] * repetitions
        l_store_detailed_counts = [store_detailed_counts] * repetitions
        l_quiet = [False] * repetitions  # Don't print info

        args = zip(l_i, l_iterations, l_study_area, l_params, l_output, l_output_every_iteration, l_use_gpu,
                   l_use_lockdown, l_start_date, l_calibration_params, l_disease_params, l_use_healthier_pop,
                   l_store_detailed_counts, l_quiet)
        to_return = None
        start_time = time.time()

        if multiprocess:
            try:
                print("Running multiple models in multiprocess mode ... ", end="", flush=True)
                with multiprocessing.Pool(processes=int(os.cpu_count())) as pool:
                    to_return = pool.starmap(OpenCLRunner.run_aspics, args)
            finally:  # Make sure they get closed (shouldn't be necessary)
                pool.close()
        else:
            results = itertools.starmap(OpenCLRunner.run_aspics, args)
            # Return as a list to force the models to execute (otherwise this is delayed because starmap returns
            # a generator. Also means we can use tqdm to get a progress bar, which is nice.
            to_return = [x for x in tqdm.tqdm(results, desc="Running models", total=repetitions)]

        print(f".. finished, took {round(float(time.time() - start_time), 2)}s)", flush=True)

        return to_return

    @classmethod
    def run_aspics_with_params_abc(cls, input_params_dict: dict, parameter_file, return_full_details=False):

        # print("opencl_runner.py -- run_model_with_params_abc")
        if not cls.initialised:
            raise Exception("The OpenCLRunner class needs to be initialised first. "
                            "Call the OpenCLRunner.init() function")

        # Check that all input parameters are not negative
        for k, v in input_params_dict.items():
            if v < 0:
                raise Exception(f"The parameter {k}={v} < 0. "
                                f"All parameters: {input_params_dict}")

        # Get help from Dustin I think this is not right.
        with open(parameter_file, "r") as f:
            parameters = load(f, Loader=SafeLoader)
        calibration_params = parameters["microsim_calibration"]
        disease_params = parameters["disease"]

        # Splat the input_params_dict to automatically set any parameters that have been inlcluded
        params = OpenCLRunner.create_params_manually(parameters_file=cls.PARAMETERS_FILE, **input_params_dict)

        # Run the `model`
        results = OpenCLRunner.run_aspics_multi(
            repetitions=cls.REPETITIONS, iterations=cls.ITERATIONS, study_area=cls.STUDY_AREA, params=params,
            output=cls.OUTPUT, output_every_iteration=cls.OUTPUT_EVERY_ITERATION, use_lockdown=cls.USE_LOCKDOWN,
            start_date=cls.START_DATE, calibration_params=calibration_params, disease_params=disease_params,
            use_gpu=cls.USE_GPU, use_healthier_pop=cls.USE_HEALTHIER_POP, store_detailed_counts=cls.STORE_DETAILED_COUNTS, multiprocess=False,
            random_ids=True)

        # Create summary objects containing the model results
        summaries = [x[0] for x in results]

        # Get the cumulative number of new infections per day (i.e. simulated results)
        sim = OpenCLRunner.get_cumulative_new_infections(summaries)
        print(f"Ran Model with {str(input_params_dict)}")

        if return_full_details:
            # Can compare these to the observations to get a fitness
            obs = cls.OBSERVATIONS.loc[:cls.ITERATIONS - 1, "Cases"].values
            assert len(sim) == len(obs)
            fitness = OpenCLRunner.fit_l2(sim, obs)
            return fitness, sim, obs, params, summaries
        else:  # Return the expected counts in a dictionary
            return {"data": sim}