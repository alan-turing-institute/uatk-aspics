/*
OpenCL Specification for futher reading and documentation:
https://registry.khronos.org/OpenCL/specs/2.2/pdf/OpenCL_C.pdf
*/

/*
  Random Number Generation
*/

// Floating point random number generation for normal and exponential variates currently uses
// Box-Muller and Inversion Transform based approaches. The ziggurat is generally preferred
// for these distributions, but we've chosen simpler, slightly more expensive options.

constant float PI = 3.14159274101257324;

// 32 bit Xoshiro128++ random number generator. Given a 128 bit uint4 state in global device
// memory, updates that state and returns a random 32 bit unsigned integer. Random states must be
// initialised externally.
uint xoshiro128pp_next(global uint4* s) {
  const uint x = s->x + s->w;
  const uint result = ((x << 7) | (x >> (32 - 7))) + s->x;

  const uint t = s->y << 9;

  s->z ^= s->x;
  s->w ^= s->y;
  s->y ^= s->z;
  s->x ^= s->w;

  s->z ^= t;
  s->w = (s->w << 11) | (s->w >> (32 - 11));

  return result;
}

// Generate a random float in the interval [0, 1]
float rand(global uint4* rng) {
  // Get the 23 upper bits (i.e number of bits in fp mantissa)
  const uint u = xoshiro128pp_next(rng) >> 9;
  // Cast to a float and divide by the largest 23 bit unsigned integer.
  return (float)u / (float)((1 << 23) - 1);
}

// Generate a sample from the standard normal distribution, calculated using the Box-Muller transform
float randn(global uint4* rng) {
  float u = rand(rng);
  float v = rand(rng);
  return sqrt(-2 * log(u)) * cos(2 * PI * v);
}

// Generate a random draw from an exponential distribution with rate 1.0 using inversion transform method.
float rand_exp(global uint4* rng) {
  return -log((float)1.0 - rand(rng));
}

// Generate a random draw from a weibull distribution with provided shape and scale
float rand_weibull(global uint4* rng, float scale, float shape) {
  return scale * pow(rand_exp(rng), ((float)1.0 / shape));
}

float lognormal(global uint4* rng, float meanlog, float sdlog){
  return exp(meanlog + sdlog * randn(rng));
}

/*
This file contains all the OpenCL kernel logic for the RAMP Urban Analytics Covid-19 model.
*/

/*
  Constants
*/

// sentinel value to indicate empty slots 
constant uint sentinel_value = ((uint)1<<31) - 1;

// Fixed point precision factor. This needs to be big enough to represent sufficiently
// small numbers (anything less than 1 / fixed_factor rounds to 0) and small enough to
// prevent overflow (anything greater than max_uint / fixed_factor will overflow).
// A good choice depends on the use, here we're mainly representing probabilities so
// this value is chosen since it matches the set of values in the unit interval
// representable by a floating point number with a fixed exponent and 23 bit significand.
constant float fixed_factor = 8388608.0;

/*
  Disease Status Enum
*/
typedef enum DiseaseStatus {
  Susceptible = 0,
  Exposed = 1,
  Presymptomatic = 2,
  Asymptomatic = 3,
  Symptomatic = 4,
  Recovered = 5,
  Dead = 6,
} DiseaseStatus;

bool is_infectious(DiseaseStatus status) {
  return status == Presymptomatic || status == Asymptomatic || status == Symptomatic;
}

/*
  Activity type enum
*/
typedef enum Activity {
  Home = 0,
  Retail = 1,
  PrimarySchool = 2,
  SecondarySchool = 3,
  Work = 4,
} Activity;


/*
  Model parameters
*/
typedef struct Params {
  float symptomatic_multiplier; // Increase in time at home if symptomatic
  float exposed_scale; // The scale of the distribution of exposed durations
  float exposed_shape; // The shape of the distribution of exposed durations
  float presymp_scale; // The scale of the distribution of presymptomatic durations
  float presymp_shape; // The shape of the distribution of presymptomatic durations
  float infection_log_scale; // The std dev of the underlying normal distribution of the lognormal infected duration distribution
  float infection_mode; // The mode of the lognormal distribution of infected durations
  float lockdown_multiplier; // Increase in time at home due to lockdown
  float place_hazard_multipliers[5]; // Hazard multipliers by activity
  float individual_hazard_multipliers[3]; // Hazard multipliers by activity
  float cvd_multiplier; // mortality multipliers for cardiovascular disease
  float diabetes_multiplier; // mortality multipliers for diabetes
  float bloodpressure_multiplier; // mortality multipliers for high blood pressure
  float health_risk_multipliers[2];
  float bmi_multipliers[12];
  float sex_multipliers[4]; 
  float ethnicity_multipliers[4];
  float age_morbidity_multipliers[9];
  float age_mortality_multipliers[9];

} Params;


// get the individual hazard multiplier for a given disease status
float get_individual_multiplier_for_status(global const struct Params* params, DiseaseStatus status) {
  // only 3 of the disease states are infections, so need to calculate the correct index into the hazard multiplier array
  int status_idx = (int)status - 2;
  return params->individual_hazard_multipliers[status_idx];
}

/*
  Utility functions
*/

// get the corresponding 1D index for a 2D index 
int get_1d_index(int person_id, int slot, int nslots) {
  return person_id*nslots + slot;
}

uint sample_exposed_duration(global uint4* rng, global const Params* params){
  return (uint)rand_weibull(rng, params->exposed_scale, params->exposed_shape);
}

uint sample_presymptomatic_duration(global uint4* rng, global const Params* params){
  return (uint)rand_weibull(rng, params->presymp_scale, params->presymp_shape);
}

uint sample_infection_duration(global uint4* rng, global const Params* params){
  float mode = params->infection_mode;
  float sdlog = params->infection_log_scale;
  float meanlog = pow(sdlog, 2) + log(mode);
  return (uint)lognormal(rng, meanlog, sdlog);
}

//NEW FUNCTION NO 1, from ratio to Prob.
float odd_ratio_to_proba (float oddRatio, float knownProb){
  return oddRatio * knownProb / (1 + oddRatio * knownProb - knownProb);
}

// NEW FUNCTION No 3, as replacement for "get_mortality_prob_for_age" including several new paramaters from SPC and the parameters file.
float get_mortality_prob_for_age(ushort age, ushort sex, int origin, ushort cvd, ushort diabetes, ushort bloodpressure, float new_bmi,  global const Params* params){
  float oddSex = ((1 - sex) * params->sex_multipliers[2]) + sex * params->sex_multipliers[0];
  float probaSex = odd_ratio_to_proba(oddSex,params->health_risk_multipliers[1]);
  float oddAge = params->age_mortality_multipliers[(int) min(age/10,8)];
  float probaAge = odd_ratio_to_proba(oddAge,probaSex);
  float oddCVD = max(cvd * params->cvd_multiplier, (float) 1.0);
  float probaCVD = odd_ratio_to_proba(oddCVD,probaAge);
  float oddDiabetes = max(diabetes * params->diabetes_multiplier, (float) 1.0);
  float probaDiabetes = odd_ratio_to_proba(oddDiabetes,probaCVD);
  float oddHypertension = max(bloodpressure * params->bloodpressure_multiplier, (float) 1.0);
  float probaHypertension = odd_ratio_to_proba(oddHypertension,probaDiabetes);
  int originNew = min(origin, 3); //origin categories 3 and 4 get merged
  float probaOrigin = odd_ratio_to_proba(params->ethnicity_multipliers[originNew],probaHypertension);
  float lower_new_bmi = 10;
  ///////////////////////
  float personal_mortality_final;
  float oddBMI;
  ///////////////////////
  
  //printf("Age_ID: %d, Origin: %d, Initial BMI: %f, Odds: %f,  Probability: %f\n", age, originNew, new_bmi, oddBMI, probaOrigin);
  
  //new_bmi = 28.0;
  //new_bmi = 40.0;
  if (new_bmi <= 0.0){ // Negatives are missing values from SPC, treat as neutral case (odds of 1)
    oddBMI = 1.0;
  } else if (new_bmi < lower_new_bmi){   ///// if New_BMI is lower than threshold, then set to threshold //////
    oddBMI = params->bmi_multipliers[originNew * 3] + params->bmi_multipliers[originNew * 3 + 1] * lower_new_bmi + params->bmi_multipliers[originNew * 3 + 2] * pown(lower_new_bmi,2);
  //} else if (new_bmi > 28.0){
  //    oddBMI = params->bmi_multipliers[originNew * 3] + params->bmi_multipliers[originNew * 3 + 1] * 0.99 * new_bmi + params->bmi_multipliers[originNew * 3 + 2] * pown(0.99 * new_bmi,2);
  //} else if (new_bmi > 29.0){
  //   oddBMI = params->bmi_multipliers[originNew * 3] + params->bmi_multipliers[originNew * 3 + 1] * (new_bmi - 1) + params->bmi_multipliers[originNew * 3 + 2] * pown(new_bmi - 1,2);
  }else {
    oddBMI = params->bmi_multipliers[originNew * 3] + params->bmi_multipliers[originNew * 3 + 1] * new_bmi + params->bmi_multipliers[originNew * 3 + 2] * pown(new_bmi,2);
  }
  //printf("Age_ID: %d, Origin: %d, Final BMI: %f, bmi_multipliers: %f,%f,%f \n", age, originNew, new_bmi, params->bmi_multipliers[originNew * 3], params->bmi_multipliers[originNew * 3 + 1], params->bmi_multipliers[originNew * 3 + 2]);
  
  personal_mortality_final = odd_ratio_to_proba(oddBMI,probaOrigin);
  //printf("Age_ID: %d, Origin: %d, Final BMI: %f, Odds: %f,  Probability: %f\n", age, originNew, new_bmi, oddBMI, personal_mortality_final);
  return personal_mortality_final;
}

//NEW FUNCTION No 2, as a replacement of "get_symptomatic_prob_for_age", where now sex is a parameter.
float get_symptomatic_prob_for_age(ushort age, ushort sex, global const Params* params){
  float oddSex = (1 - sex) * params->sex_multipliers[3] + sex * params->sex_multipliers[1];
  float probaSex = odd_ratio_to_proba(oddSex,params->health_risk_multipliers[0]);
  float oddAge = params->age_morbidity_multipliers[(int) min(age/10,8)];
  float personal_morbidity_final = odd_ratio_to_proba(oddAge,probaSex);
  return personal_morbidity_final;
} 


/*
  Kernels
*/

// Reset the hazard and count of each place to zero.
kernel void places_reset(uint nplaces,
                         global uint* place_hazards,
                         global uint* place_counts) {
  int place_id = get_global_id(0);
  if (place_id >= nplaces) return;

  place_hazards[place_id] = 0;
  place_counts[place_id] = 0;
}

// Compute and set the movement flows for all the places for each person, 
// given the person's baseline movement flows (pre-calculated from activity specific flows and durations) and disease status.
// Includes lockdown logic.
kernel void people_update_flows(uint npeople,
                                uint nslots,
                                global const uint* people_statuses,
                                global const float* people_flows_baseline,
                                global float* people_flows,
                                global const uint* people_place_ids,
                                global const uint* place_activities,
                                global const struct Params* params) {
  int person_id = get_global_id(0);
  if (person_id >= npeople) return;

  uint person_status = people_statuses[person_id];

  // choose flow multiplier based on whether person is symptomatic or not
  // NB: lockdown is assumed not to change behaviour of symptomatic people, since it will already be reduced
  float non_home_multiplier = ((DiseaseStatus)person_status == Symptomatic) ? params->symptomatic_multiplier : params->lockdown_multiplier;
  
  float total_new_flow = 0.0;
  uint home_flow_idx = 0;
  
  // adjust non-home activity flows by the chosen multiplier, while summing the new flows so we can calculate the new home flow
  for(int slot = 0; slot < nslots; slot++){
    uint flow_idx = get_1d_index(person_id, slot, nslots);
    float baseline_flow = people_flows_baseline[flow_idx];
    uint place_id = people_place_ids[flow_idx];

    // check it is not an empty slot
    if (place_id != sentinel_value){
      uint activity = (Activity)place_activities[place_id];
      if (activity == Home) {
        // store flow index of home 
        home_flow_idx = flow_idx;
      } else { 
        // for non-home activities - adjust flow by multiplier
        float new_flow = baseline_flow * non_home_multiplier;
        people_flows[flow_idx] = new_flow;
        total_new_flow += new_flow;
      }
    }
  }

  // new home flow is 1 minus the total new flows for non-home activities, since all flows should sum to 1, 
  people_flows[home_flow_idx] = 1.0 - total_new_flow;
}

// Given their current status, accumulate hazard from each person into their candidate places.
kernel void people_send_hazards(uint npeople,
                                uint nslots,
                                global const uint* people_statuses,
                                global const uint* people_place_ids,
                                global const float* people_flows,
                                global const float* people_hazards,
                                volatile global uint* place_hazards,
                                volatile global uint* place_counts,
                                global const uint* place_activities,
                                global const Params* params) {
  int person_id = get_global_id(0);
  if (person_id >= npeople) return;

  // Early return for non infectious people
  DiseaseStatus person_status = (DiseaseStatus)people_statuses[person_id];
  if (!is_infectious(person_status)) return;

  for (int slot=0; slot < nslots; slot++) {
    // Get the place and flow for this slot
    uint flow_idx = get_1d_index(person_id, slot, nslots);
    uint place_id = people_place_ids[flow_idx];

    //check it is not an empty slot
    if (place_id == sentinel_value) continue;

    float flow = people_flows[flow_idx];
    uint activity = place_activities[place_id];

    //check it is a valid activity and select hazard multiplier
    float place_multiplier = (0 <= activity && activity <= 4) ? params->place_hazard_multipliers[activity] : 1.0;
    float individual_multiplier = get_individual_multiplier_for_status(params, person_status);

    float hazard_increase = flow * place_multiplier * individual_multiplier;

    // Convert the flow to fixed point
    uint fixed_hazard_increase = (uint)(fixed_factor * hazard_increase);

    // Atomically add hazard increase and increment counts for this place
    atomic_add(&place_hazards[place_id], fixed_hazard_increase);
    atomic_add(&place_counts[place_id], 1);
  }
}

//For each person accumulate hazard from all the places stored in their slots.
kernel void people_recv_hazards(uint npeople,
                                uint nslots,
                                global const uint* people_statuses,
                                global const uint* people_place_ids,
                                global const float* people_flows,
                                global float* people_hazards,
                                global const uint* place_hazards,
                                global const Params* params) {
  int person_id = get_global_id(0);
  if (person_id >= npeople) return;

  // Early return for non susceptible people
  DiseaseStatus person_status = (DiseaseStatus)people_statuses[person_id];
  if (person_status != Susceptible) return;

  // Initialize hazard to accumulate into
  float hazard = 0.0;

  for (int slot=0; slot < nslots; slot++) {
    // Get the place and flow for this slot
    uint flow_idx = get_1d_index(person_id, slot, nslots);
    uint place_id = people_place_ids[flow_idx];
    
    //check it is not an empty slot
    if (place_id == sentinel_value) continue;

    float flow = people_flows[flow_idx];

    // Get the hazard and convert it to floating point
    uint fixed_hazard = place_hazards[place_id];
    hazard += flow * (float)fixed_hazard / fixed_factor;
  }

  // Write the total hazard onto the individual
  people_hazards[person_id] = hazard;
}

// Disease model: given their current disease status and hazard, determine if a person is due to transition to the next
// state, and if so apply that transition.
kernel void people_update_statuses(uint npeople,
                                   global const ushort* people_ages,
                                   global const float* people_new_bmi,
                                   global const ushort* people_obesity,
                                   global const uchar* people_cvd,
                                   global const uchar* people_diabetes,
                                   global const uchar* people_bloodpressure,
                                   global const ushort* people_sex,
                                   global const ushort* people_origin,
                                   global const float* people_hazards,
                                   global uint* people_statuses,
                                   global uint* people_transition_times,
                                   global uint4* people_prngs,
                                   global const Params* params) {
  int person_id = get_global_id(0);
  if (person_id >= npeople) return;

  global uint4* rng = &people_prngs[person_id];

  DiseaseStatus current_status = (DiseaseStatus)people_statuses[person_id];
  DiseaseStatus next_status = current_status;

  uint current_transition_time = people_transition_times[person_id];
  uint next_transition_time = current_transition_time;

  // assign new infections to susceptible people 
  if (current_status == Susceptible){
    float hazard = people_hazards[person_id];
    // Integrate hazard into probability
    float infection_prob = 1.0 - exp(-hazard);

    // randomly sample if they should be infected or not based on infection probability
    if (rand(rng) < infection_prob) {
      next_status = Exposed;
      next_transition_time = sample_exposed_duration(rng, params);
    }
  }

  // cycle through disease states
  if( current_transition_time <= 0 ) { // if time to transition to next state
    switch(current_status) {
        case Exposed:
        {
          ushort person_age = people_ages[person_id];
          ushort person_sex = people_sex[person_id];

          //Calling FUNCTION No 3, as a replace of "get_symptomatic_prob_for_age", where now sex is a parameter.
          float symptomatic_prob = get_symptomatic_prob_for_age(person_age, person_sex, params);
          
          // randomly select whether to become asymptomatic or presymptomatic
          next_status = rand(rng) < symptomatic_prob ? Presymptomatic : Asymptomatic;
          
          //choose transition time based on presymptomatic or asymptomatic
          next_transition_time = next_status == Presymptomatic ? sample_presymptomatic_duration(rng, params) : sample_infection_duration(rng, params);
          
          break;
        }
        case Presymptomatic:
        {
          next_status = Symptomatic;
          next_transition_time = sample_infection_duration(rng, params);
          break;
        }
        case Symptomatic:
        {
          // Calculate recovered prob based on age
          ushort person_age = people_ages[person_id];
          ushort person_sex = people_sex[person_id];
          ushort person_origin = people_origin[person_id];
          float person_new_bmi = people_new_bmi[person_id];
          ushort person_cvd = people_cvd[person_id];
          ushort person_diabetes = people_diabetes[person_id];
          ushort person_bloodpressure = people_bloodpressure[person_id];

          float mortality_prob = get_mortality_prob_for_age(person_age, person_sex,person_origin, person_cvd, person_diabetes, person_bloodpressure, person_new_bmi, params);

          // randomly select whether dead or recovered
          next_status = rand(rng) > mortality_prob ? Recovered : Dead;
          break;
        }
        case Asymptomatic:
        {
          next_status = Recovered; //assuming all asymptomatic infections recover
          break;
        }
        default:
          break;
    }
  }
  
  // decrement transition time each timestep
  if(next_transition_time > 0){
    next_transition_time--;
  }

  // apply new statuses and transition times
  people_statuses[person_id] = next_status;
  people_transition_times[person_id] = next_transition_time;
}
