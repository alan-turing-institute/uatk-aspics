

def personal_morbidity (age, sex, Params(sex_parameters, age_parameters)):
    personal_morbidity = ((1- sex)* params["female_symptomatic"] + sex * params[male_symptomatic])*  symptomatic_age[min(age/bin_size, max_bin_idx)]
return personal_morbidity



def personal_mortality(age, sex,origin, cvd, diabetes, bloodpresure, bmiNew, paramas(health_conditions):
    result1 = ((1- sex)* params["female_mortality"] + sex * params["male_mortality"])* mortality_age[min(age/bin_size, max_bin_idx)]
    result2 = max (cvd * health_conditions["cvd"], 1)  * max (diabetes * health_conditions["diabetes"], 1) * max (bloodpresure * health_conditions["bloodpresure"], 1)
    #array with the 12 categories from BIM --> BMI_types
    origin = min (origin, 4)  # BMI data 4 and 5 are merged.
    bmi_final = global_bmi * (BMI_types [(origin -1) *3] + BMI_types [(origin -1) *3 + 1] * age + BMI_types [(origin -1) *3 +2]* age ^2)
return mortality_risk=bmi_final * result1 * result2