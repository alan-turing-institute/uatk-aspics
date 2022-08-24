import math

param_morbidity = params["morbidity"]
param_age_morbidity = [params["a0-9_morbidity"],params["a10-19_morbidity"],params["a20-29_morbidity"],params["a30-39_morbidity"],params["a40-49_morbidity"],
                       params["a50-59_morbidity"],params["a60-69_morbidity"],params["a70-79_morbidity"],params["a80plus_morbidity"]
                      ]
param_age_mortality = [params["a0-9_mortality"],params["a10-19_mortality"],params["a20-29_mortality"],params["a30-39_mortality"],params["a40-49_mortality"],
                       params["a50-59_mortality"],params["a60-69_mortality"],params["a70-79_mortality"],params["a80plus_mortality"]
                      ]
param_bmi_mortality = [params["white_Ethni_coef0"],params["white_Ethni_coef1"],params["white_Ethni_coef2"],params["black_Ethni_coef0"],
                       params["black_Ethni_coef1"],params["black_Ethni_coef2"],params["asian_Ethni_coef0"],params["asian_Ethni_coef1"],
                       params["asian_Ethni_coef2"],params["other_Ethni_coef0"],params["other_Ethni_coef1"],params["other_Ethni_coef2"]
                      ]

def odd_ratio_to_proba(oddRatio,knownProb):
    return oddRatio * knownProb / (1 + oddRatio * knownProb - knownProb)

def personal_morbidity(age, sex, params, param_morbidity, param_age_morbidity):
    oddSex = (1 - sex) * params["female_symptomatic"] + sex * params["male_symptomatic"]
    probaSex = odd_ratio_to_proba(oddSex,param_morbidity)
    oddAge = param_age_morbidity[min(math.floor(age/10),8)]
    personal_morbidity_final = odd_ratio_to_proba(oddAge,probaSex)
    return personal_morbidity_final

def personal_mortality(age, sex, origin, cvd, diabetes, bloodpressure, bmiNew, params, param_mortality, param_age_mortalit, param_bmi_mortality):
    oddSex = (1 - sex) * params["female_mortality"] + sex * params["male_mortality"]
    probaSex = odd_ratio_to_proba(oddSex,param_mortality)
    oddAge = param_age_mortality[min(math.floor(age/10),8)]
    probaAge = odd_ratio_to_proba(oddAge,probaSex)
    oddCVD = max(cvd * params["cvd_mortality"], 1.0)
    probaCVD = odd_ratio_to_proba(oddCVD,probaAge)
    oddDiabetes = max(diabetes * params["diabetes_mortality"], 1.0)
    probaDiabetes = odd_ratio_to_proba(oddDiabetes,probaCVD)
    oddHypertension = max(bloodpressure * params["bloodpresure_mortality"], 1.0)
    probaHypertension = odd_ratio_to_proba(oddHypertension,probaDiabetes)
    oddOrigin = [params["white_mortality"],params["white_mortality"],params["white_mortality"],params["white_mortality"]]
    originNew = min (origin, 4)  # BMI data 4 and 5 get merged
    probaOrigin = odd_ratio_to_proba(oddOrigin[originNew - 1],probaHypertension)
    oddBMI = (param_bmi_mortality[(originNew -1) * 3] + param_bmi_mortality[(originNew -1) * 3 + 1] * bmiNew + param_bmi_mortality[(originNew -1) * 3 + 2]* bmiNew ^ 2)
    personal_morbidity_final = odd_ratio_to_proba(oddBMI,probaOrigin)
    return personal_morbidity_final
