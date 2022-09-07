import numpy as np
from collections import namedtuple

LocationHazardMultipliers = namedtuple(
    "LocationHazardMultipliers",
    ["retail", "primary_school", "secondary_school", "home", "work"],
)

IndividualHazardMultipliers = namedtuple(
    "IndividualHazardMultipliers", ["presymptomatic", "asymptomatic", "symptomatic"]
)


class Params:
    """Convenience class for setting simulator parameters. Also holds the default values."""

    def __init__(
            self,
            ### PLACES#######
            ## TODO Ask Dustin why this is also hardcoded. 
            location_hazard_multipliers=LocationHazardMultipliers(
                retail=0.0165,
                primary_school=0.0165,
                secondary_school=0.0165,
                home=0.0165,
                work=0.0,
            ),
            ### DISEASE STATUS##########
            ## TODO Ask Dustin why this is also hardcoded. 
            individual_hazard_multipliers=IndividualHazardMultipliers(
                presymptomatic=1.0, asymptomatic=0.75, symptomatic=1.0
            ),
            ###########################################
            #####Health Conditions (Type, BMI) ########
            ###########################################
            sex_multipliers = [1.0,1.0,1.0,1.0],
            ethnicity_multipliers =[1.0,1.0,1.0,1.0],
            age_morbidity_multipliers = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
            age_mortality_multipliers = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
            cvd_multiplier=1.0,
            diabetes_multiplier=1.0,
            bloodpressure_multiplier=1.0,
            health_risk_multipliers=[1.0, 1.0],
            bmi_multipliers=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],

    ):
        """Create a simulator with the default parameters."""
        if health_risk_multipliers is None:
             health_risk_multipliers = [1.0, 1.0]

        if bmi_multipliers is None:
            bmi_multipliers = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        if sex_multipliers is None:
            sex_multipliers = [1,1.34,1,1.0]

        ## TODO Ask Hadried/Dustin why this is needed. I guess with the new version can be removed.
        self.symptomatic_multiplier = 0.5

        ## TODO Ask Dustin why this is also hardcoded. 
        ### Paramaters for the GUI####
        self.exposed_scale = 2.82
        self.exposed_shape = 3.99

        self.presymptomatic_scale = 2.45
        self.presymptomatic_shape = 7.79

        self.infection_log_scale = 0.35
        self.infection_mode = 7.0

        ### Paramaters for the Sumulation####
        self.lockdown_multiplier = 1.0
        self.place_hazard_multipliers = np.array(
            [
                location_hazard_multipliers.retail,
                location_hazard_multipliers.primary_school,
                location_hazard_multipliers.secondary_school,
                location_hazard_multipliers.home,
                location_hazard_multipliers.work,
            ],
            dtype=np.float32,
        )

        self.individual_hazard_multipliers = np.array(
            [
                individual_hazard_multipliers.presymptomatic,
                individual_hazard_multipliers.asymptomatic,
                individual_hazard_multipliers.symptomatic,
            ],
            dtype=np.float32,
        )

        self.cvd_multiplier = cvd_multiplier
        self.diabetes_multiplier = diabetes_multiplier
        self.bloodpressure_multiplier = bloodpressure_multiplier
        self.health_risk_multipliers = health_risk_multipliers
        self.bmi_multipliers = bmi_multipliers
        self.sex_multipliers = sex_multipliers
        self.ethnicity_multipliers = ethnicity_multipliers
        self.age_mortality_multipliers = age_mortality_multipliers
        self.age_morbidity_multipliers = age_morbidity_multipliers

    def asarray(self):
        """Pack the parameters into a flat array for uploading."""
        return np.concatenate(
            [
                np.array(
                    [
                        self.symptomatic_multiplier,
                        self.exposed_scale,
                        self.exposed_shape,
                        self.presymptomatic_scale,
                        self.presymptomatic_shape,
                        self.infection_log_scale,
                        self.infection_mode,
                        self.lockdown_multiplier,
                    ],
                    dtype=np.float32,
                ),
                self.place_hazard_multipliers,
                self.individual_hazard_multipliers,
                np.array(
                    [
                        self.cvd_multiplier,
                        self.diabetes_multiplier,
                        self.bloodpressure_multiplier,
                    ],
                    dtype=np.float32,
                ),
                self.health_risk_multipliers,
                self.bmi_multipliers,
                self.sex_multipliers,
                self.ethnicity_multipliers,
                self.age_morbidity_multipliers,
                self.age_mortality_multipliers
            ]
        )

    @classmethod
    def fromarray(cls, params_array):
        location_hazard_multipliers = LocationHazardMultipliers(
            retail=params_array[8],
            primary_school=params_array[9],
            secondary_school=params_array[10],
            home=params_array[11],
            work=params_array[12],
        )
        individual_hazard_multipliers = IndividualHazardMultipliers(
            presymptomatic=params_array[13],
            asymptomatic=params_array[14],
            symptomatic=params_array[15],
        )
        p = cls(location_hazard_multipliers, individual_hazard_multipliers)
        p.symptomatic_multiplier = params_array[0]
        p.exposed_scale = params_array[1]
        p.exposed_shape = params_array[2]
        p.presymptomatic_scale = params_array[3]
        p.presymptomatic_shape = params_array[4]
        p.infection_log_scale = params_array[5]
        p.infection_mode = params_array[6]
        p.lockdown_multiplier = params_array[7]
        p.cvd_multiplier = params_array[16]
        p.diabetes_multiplier = params_array[17]
        p.bloodpressure_multiplier = params_array[18]
        p.health_risk_multipliers = params_array[19:21]
        p.bmi_multipliers = params_array[21:33]
        p.sex_multipliers = params_array[33:37]
        p.ethnicity_multipliers = params_array[37:41]
        p.age_morbidity_multipliers=params_array[41:50]
        p.age_mortality_multipliers=params_array[50:60]
        
        return p

    def set_lockdown_multiplier(self, lockdown_multipliers, timestep):
        """Update the lockdown multiplier based on the current time."""
        self.lockdown_multiplier = lockdown_multipliers[
            np.minimum(lockdown_multipliers.shape[0] - 1, timestep)
        ]

    def num_bytes(self):
        return 4 * self.asarray().size
