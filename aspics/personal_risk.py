import math

def personal_morbidity(age, sex, params):
    param_age_morbidity = [params["a0-9_morbidity"],params["a10-19_morbidity"],params["a20-29_morbidity"],params["a30-39_morbidity"],params["a40-49_morbidity"],
                           params["a50-59_morbidity"],params["a60-69_morbidity"],params["a70-79_morbidity"],params["a80plus_morbidity"]
                          ]
    personal_morbidity_final = params["morbidity"] * ((1 - sex) * params["female_symptomatic"] + sex * params["male_symptomatic"]) * param_age_morbidity[min(math.floor(age/10),8)]
    return personal_morbidity_final

def personal_mortality(age, sex, origin, cvd, diabetes, bloodpressure, bmiNew, params):
    param_age_mortality = [params["a0-9_mortality"],params["a10-19_mortality"],params["a20-29_mortality"],params["a30-39_mortality"],params["a40-49_mortality"],
                           params["a50-59_mortality"],params["a60-69_mortality"],params["a70-79_mortality"],params["a80plus_mortality"]
                          ]
    sexAndAge = ((1 - sex) * params["female_mortality"] + sex * params["male_mortality"]) * param_age_mortality[min(math.floor(age/10), 8)]
    healthCondition = max (cvd * params["cvd_mortality"], 1.0)  * max (diabetes * params["diabetes_mortality"], 1.0) * max (bloodpressure * params["bloodpresure_mortality"], 1.0)
    param_origin_mortality = [params["white_mortality"],params["white_mortality"],params["white_mortality"],params["white_mortality"]]
    param_bmi_mortality = [params["white_Ethni_coef0"],params["white_Ethni_coef1"],params["white_Ethni_coef2"],params["black_Ethni_coef0"],
                           params["black_Ethni_coef1"],params["black_Ethni_coef2"],params["asian_Ethni_coef0"],params["asian_Ethni_coef1"],
                           params["asian_Ethni_coef2"],params["other_Ethni_coef0"],params["other_Ethni_coef1"],params["other_Ethni_coef2"]
                          ]
    originNew = min (origin, 4)  # BMI data 4 and 5 get merged
    originAndBMI = (param_origin_mortality[originNew - 1]) * (param_bmi_mortality[(originNew -1) * 3] + param_bmi_mortality[(originNew -1) * 3 + 1] * bmiNew + param_bmi_mortality[(originNew -1) * 3 + 2]* bmiNew ^ 2)
    personal_morbidity_final = params["mortality"] * sexAndAge * healthCondition * originAndBMI
    return personal_morbidity_final