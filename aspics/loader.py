import os
import numpy as np
from yaml import load, SafeLoader

from aspics.simulator import Simulator
from aspics.snapshot import Snapshot
from aspics.params import Params, IndividualHazardMultipliers, LocationHazardMultipliers


def setup_sim_from_file(parameters_file):
    print(f"Running a simulation based on {parameters_file}")
    with open(parameters_file, "r") as f:
        parameters = load(f, Loader=SafeLoader)
        return setup_sim(parameters)


def setup_sim(parameters):
    print(f"Running a manually added parameters simulation based on {parameters}")

    sim_params = parameters["microsim"] ## Set of Parameters for the ASPCIS microsim
    calibration_params = parameters["microsim_calibration"] ## Calibration paramaters
    disease_params = parameters["disease"] # Disease paramaters for the moment only beta_risk included.
    health_conditions = parameters["health_conditions"] # individual health conditions
    iterations = sim_params["iterations"]
    study_area = sim_params["study-area"]
    output = sim_params["output"]
    output_every_iteration = sim_params["output-every-iteration"]
    use_lockdown = sim_params["use-lockdown"]
    start_date = sim_params["start-date"]

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

    # TODO set the random seed of the model. Why do we still have this very random number
    snapshot.seed_prngs(42)

    # set params
    if calibration_params is not None and disease_params is not None and health_conditions is not None:
        snapshot.update_params(create_params(calibration_params, disease_params, health_conditions))
        #snapshot.change_neg_values_new_bmi()

    # Create a simulator and upload the snapshot data to the OpenCL device
    simulator = Simulator(snapshot, study_area, gpu=True)
    [people_statuses, people_transition_times] = simulator.seeding_base()
    simulator.upload_all(snapshot.buffers)
    simulator.upload("people_statuses", people_statuses)
    simulator.upload("people_transition_times", people_transition_times)

    return simulator, snapshot, study_area, iterations


def create_params(calibration_params, disease_params, health_conditions):

    current_risk_beta = disease_params["current_risk_beta"]
    # NB: OpenCL model incorporates the current risk beta by pre-multiplying the hazard multipliers with it
    location_hazard_multipliers = LocationHazardMultipliers(
        retail=calibration_params["hazard_location_multipliers"]["Retail"]
               * current_risk_beta,
        primary_school=calibration_params["hazard_location_multipliers"][
                           "PrimarySchool"
                       ]
                       * current_risk_beta,
        secondary_school=calibration_params["hazard_location_multipliers"][
                             "SecondarySchool"
                         ]
                         * current_risk_beta,
        home=calibration_params["hazard_location_multipliers"]["Home"]
             * current_risk_beta,
        work=calibration_params["hazard_location_multipliers"]["Work"]
             * current_risk_beta,
    )

    individual_hazard_multipliers = IndividualHazardMultipliers(
        presymptomatic=calibration_params["hazard_individual_multipliers"][
            "presymptomatic"
        ],
        asymptomatic=calibration_params["hazard_individual_multipliers"][
            "asymptomatic"
        ],
        symptomatic=calibration_params["hazard_individual_multipliers"]["symptomatic"],
    )
    
    health_risk_multipliers = [
        health_conditions["global"]["morbidity"],
        health_conditions["global"]["mortality"],
    ]

    bmi_multipliers = [
        health_conditions["BMI"]["white_Ethni_coef0"],
        health_conditions["BMI"]["white_Ethni_coef1"],
        health_conditions["BMI"]["white_Ethni_coef2"],
        health_conditions["BMI"]["black_Ethni_coef0"],
        health_conditions["BMI"]["black_Ethni_coef1"],
        health_conditions["BMI"]["black_Ethni_coef2"],
        health_conditions["BMI"]["asian_Ethni_coef0"],
        health_conditions["BMI"]["asian_Ethni_coef1"],
        health_conditions["BMI"]["asian_Ethni_coef2"],
        health_conditions["BMI"]["other_Ethni_coef0"],
        health_conditions["BMI"]["other_Ethni_coef1"],
        health_conditions["BMI"]["other_Ethni_coef2"],
    ]

    sex_multipliers =[
        health_conditions["sex"]["male_mortality"],
        health_conditions["sex"]["male_symptomatic"],
        health_conditions["sex"]["female_mortality"],
        health_conditions["sex"]["female_symptomatic"],
    ]

    ethnicity_multipliers =[
        health_conditions["ethnicity"]["white_mortality"],
        health_conditions["ethnicity"]["black_mortality"],
        health_conditions["ethnicity"]["asian_mortality"],
        health_conditions["ethnicity"]["other_mortality"],
    ]

    age_morbidity_multipliers = [
        health_conditions["age_morbidity"]["a0-9_morbidity"],
        health_conditions["age_morbidity"]["a10-19_morbidity"],
        health_conditions["age_morbidity"]["a20-29_morbidity"],
        health_conditions["age_morbidity"]["a30-39_morbidity"],
        health_conditions["age_morbidity"]["a40-49_morbidity"],
        health_conditions["age_morbidity"]["a50-59_morbidity"],
        health_conditions["age_morbidity"]["a60-69_morbidity"],
        health_conditions["age_morbidity"]["a70-79_morbidity"],
        health_conditions["age_morbidity"]["a80plus_morbidity"],
    ]

    age_mortality_multipliers =[
        health_conditions["age_mortality"]["a0-9_mortality"],
        health_conditions["age_mortality"]["a10-19_mortality"],
        health_conditions["age_mortality"]["a20-29_mortality"],
        health_conditions["age_mortality"]["a30-39_mortality"],
        health_conditions["age_mortality"]["a40-49_mortality"],
        health_conditions["age_mortality"]["a50-59_mortality"],
        health_conditions["age_mortality"]["a60-69_mortality"],
        health_conditions["age_mortality"]["a70-79_mortality"],
        health_conditions["age_mortality"]["a80plus_mortality"],
        ]

    return Params(
        location_hazard_multipliers=location_hazard_multipliers,
        individual_hazard_multipliers=individual_hazard_multipliers,
        cvd_multiplier=health_conditions["type"]["cvd"],
        diabetes_multiplier=health_conditions["type"]["diabetes"],
        bloodpressure_multiplier=health_conditions["type"]["bloodpressure"],
        health_risk_multipliers = health_risk_multipliers,
        bmi_multipliers=bmi_multipliers,
        sex_multipliers = sex_multipliers,
        ethnicity_multipliers = ethnicity_multipliers,
        age_morbidity_multipliers = age_morbidity_multipliers,
        age_mortality_multipliers = age_mortality_multipliers,
    )
