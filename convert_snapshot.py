#!/usr/bin/env python3

import click
import numpy as np
from aspics.params import Params
from tqdm import tqdm
from linetimer import CodeTimer
import os
import pathlib
import random
import synthpop_pb2
from collections import namedtuple

# Things to keep in mind:
# - This script isn't deterministic, because it doesn't fix a random number
#   generator seed. We'd need to set one both from random.choice and numpy.


@click.command()
@click.option(
    "-i",
    "--input_path",
    type=click.Path(exists=True),
    help="Convert this SPC protobuf file",
)
@click.option(
    "-o", "--output_path", type=click.Path(), help="Write the snapshot file here"
)
def main(input_path, output_path):
    try:
        pathlib.Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass

    print(f"Reading {input_path}")
    with CodeTimer(unit="s"):
        pop = synthpop_pb2.Population()
        f = open(input_path, "rb")
        pop.ParseFromString(f.read())
        f.close()

    convert_to_npz(pop, output_path)


# A slot is a place somebody could visit
SLOTS = 16


class IDMapping:
    """
    Maps an activity and venue ID to a global place ID, which represent every
    possible place in the model.
    """

    def __init__(self, pop):
        self.total_places = sum(
            [len(venues.venues) for venues in pop.venues_per_activity.values()]
        ) + len(pop.households)
        # Per place, the activity associated with it
        self.place_activities = np.zeros(self.total_places, dtype=np.uint32)
        self.id_offset_per_activity = dict()

        offset = 0
        for activity in synthpop_pb2.Activity.values():
            self.id_offset_per_activity[activity] = offset
            if activity == synthpop_pb2.Activity.HOME:
                num_venues = len(pop.households)
            else:
                num_venues = len(pop.venues_per_activity[activity].venues)

            start = offset
            offset = offset + num_venues
            self.place_activities[start:offset] = activity
        assert offset == self.total_places, f"{offset} vs {self.total_places}"

    def to_place(self, activity, venue):
        return self.id_offset_per_activity[activity] + venue


def convert_to_npz(pop, output_path):
    remove_large_households(pop)

    id_mapping = IDMapping(pop)
    num_people = len(pop.people)
    num_places = id_mapping.total_places

    print(f"Collapsing flows for {len(pop.people)} people")
    with CodeTimer(unit="s"):
        people_place_ids, people_baseline_flows = get_baseline_flows(pop, id_mapping)
    print("Finalizing all coordinates")
    with CodeTimer(unit="s"):
        place_coords = get_place_coordinates(pop, id_mapping)

    print("Creating snapshot")
    with CodeTimer(unit="s"):
        np.savez(
            output_path,
            nplaces=np.uint32(num_places),
            npeople=np.uint32(num_people),
            nslots=np.uint32(SLOTS),
            time=np.uint32(0),
            not_home_probs=np.array([p.time_use.not_home for p in pop.people]),
            lockdown_multipliers=np.array(pop.lockdown.per_day, dtype=np.float32),
            place_activities=id_mapping.place_activities,
            place_coords=place_coords,
            place_hazards=np.zeros(num_places, dtype=np.uint32),
            place_counts=np.zeros(num_places, dtype=np.uint32),
            people_ages=np.array(
                [p.demographics.age_years for p in pop.people], dtype=np.uint16
            ),
            people_obesity=np.array(
                [obesity_value(p.health.bmi) for p in pop.people], dtype=np.uint16
            ),
            people_cvd=np.array(
                [bool_to_int(p.health.has_cardiovascular_disease) for p in pop.people],
                dtype=np.uint8,
            ),
            people_diabetes=np.array(
                [bool_to_int(p.health.has_diabetes) for p in pop.people], dtype=np.uint8
            ),
            people_blood_pressure=np.array(
                [bool_to_int(p.health.has_high_blood_pressure) for p in pop.people],
                dtype=np.uint8,
            ),
            people_statuses=np.zeros(num_people, dtype=np.uint32),
            people_transition_times=np.zeros(num_people, dtype=np.uint32),
            people_place_ids=people_place_ids,
            people_baseline_flows=people_baseline_flows,
            people_flows=people_baseline_flows,
            people_hazards=np.zeros(num_people, dtype=np.uint32),
            people_prngs=np.random.randint(
                np.uint32((1 << 32) - 1), size=num_people * 4, dtype=np.uint32
            ),
            area_codes=np.array(
                [pop.households[p.household].msoa11cd for p in pop.people]
            ),
            params=Params().asarray(),
        )
    print(f"Wrote {output_path}")


def get_baseline_flows(pop, id_mapping):
    # We ultimately want a 1D array for flows and place IDs. It's a flattened list, with
    # places_to_keep_per_person entries per person.
    places_to_keep_per_person = SLOTS

    sentinel_value = (1 << 31) - 1
    people_place_ids = np.full(
        len(pop.people) * places_to_keep_per_person, sentinel_value, dtype=np.uint32
    )
    people_baseline_flows = np.zeros(
        len(pop.people) * places_to_keep_per_person, dtype=np.float32
    )

    for person in tqdm(pop.people):
        idx = person.id * places_to_keep_per_person
        # Per person, flatten all the flows, regardless of activity
        for (activity, venue, weight) in get_baseline_flows_per_person(
            pop, person, places_to_keep_per_person
        ):
            people_place_ids[idx] = id_mapping.to_place(activity, venue)
            people_baseline_flows[idx] = weight
            idx += 1

    return (people_place_ids, people_baseline_flows)


def get_baseline_flows_per_person(pop, person, places_to_keep_per_person):
    result = []

    # Home and work are per-person
    result.append((synthpop_pb2.Activity.HOME, person.household, 1.0))
    if person.workplace != 2**64 - 1:
        result.append((synthpop_pb2.Activity.WORK, person.workplace, 1.0))

    # Build a map from activity to duration
    activity_durations = dict(
        [(x.activity, x.duration) for x in person.activity_durations]
    )

    # The other flows are the same for everyone in the MSOA
    msoa = pop.households[person.household].msoa11cd
    for flows in pop.info_per_msoa[msoa].flows_per_activity:
        # Weight the per-activity flow by duration
        activity_duration = activity_durations[flows.activity]
        for flow in flows.flows:
            weight = activity_duration * flow.weight
            result.append((flows.activity, flow.venue_id, weight))

    # Sort by flows, descending
    result.sort(reverse=True, key=lambda pair: pair[2])
    # Only keep the top few
    del result[places_to_keep_per_person:]
    return result


def get_place_coordinates(pop, id_mapping):
    result = np.zeros(id_mapping.total_places * 2, dtype=np.float32)

    for activity, venues in pop.venues_per_activity.items():
        # Not stored as venues
        if activity == synthpop_pb2.Activity.HOME:
            continue
        for venue in venues.venues:
            place = id_mapping.to_place(activity, venue.id)
            result[place * 2] = venue.location.latitude
            result[place * 2 + 1] = venue.location.longitude

    # For homes, we just pick a random building in the MSOA area. This is just
    # used for visualization, so lack of buildings mapped in some areas isn't
    # critical.
    for household in pop.households:
        place = id_mapping.to_place(synthpop_pb2.Activity.HOME, household.id)
        # Every MSOA is guaranteed to have buildings
        location = random.choice(pop.info_per_msoa[household.msoa11cd].buildings)
        result[place * 2] = location.latitude
        result[place * 2 + 1] = location.longitude

    return result


def bool_to_int(x):
    if x:
        return 1
    else:
        return 0


def obesity_value(x):
    # The protobuf enum defines NORMAL as 2. We want to treat NOT_APPLICABLE and UNDERWEIGHT as NORMAL.
    return max(0, x - 2)


def remove_large_households(pop, max_people_per_household=10):
    """
    Removes people living in households with too many members.

    Note this breaks some invariants of the Population data, because it does
    not reassign household and person IDs. This is sufficient for this
    particular script, but be careful if adapting this approach.
    """

    large_households = set()
    people_removed = 0
    for household in pop.households:
        if len(household.members) > max_people_per_household:
            large_households.add(household.id)
            people_removed += len(household.members)

    print(
        f"Removing {people_removed} people from {len(large_households)} households because the household has > {max_people_per_household} people"
    )

    # These aren't normal Python lists; see
    # https://developers.google.com/protocol-buffers/docs/reference/python-generated#repeated-message-fields.
    # It unfortunately copies the messages.
    people = list(
        filter(lambda person: person.household not in large_households, pop.people)
    )
    del pop.people[:]
    pop.people.extend(people)

    # Reassign people IDs, since we removed many entries from the middle of the list
    for idx, person in enumerate(pop.people):
        person.id = idx

    # If we needed to, we could fix the IDs in household.members. We could also
    # remove households and fix up person.household. But that's not necessary
    # for this script.


if __name__ == "__main__":
    main()
