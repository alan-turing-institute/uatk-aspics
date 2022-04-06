# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: synthpop.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='synthpop.proto',
  package='synthpop',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0esynthpop.proto\x12\x08synthpop\"\xf9\x02\n\nPopulation\x12\'\n\nhouseholds\x18\x01 \x03(\x0b\x32\x13.synthpop.Household\x12 \n\x06people\x18\x02 \x03(\x0b\x32\x10.synthpop.Person\x12H\n\x13venues_per_activity\x18\x03 \x03(\x0b\x32+.synthpop.Population.VenuesPerActivityEntry\x12<\n\rinfo_per_msoa\x18\x04 \x03(\x0b\x32%.synthpop.Population.InfoPerMsoaEntry\x1aM\n\x16VenuesPerActivityEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\"\n\x05value\x18\x02 \x01(\x0b\x32\x13.synthpop.VenueList:\x02\x38\x01\x1aI\n\x10InfoPerMsoaEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.synthpop.InfoPerMSOA:\x02\x38\x01\"H\n\tHousehold\x12\n\n\x02id\x18\x01 \x01(\x04\x12\x0c\n\x04msoa\x18\x02 \x01(\t\x12\x10\n\x08orig_hid\x18\x03 \x01(\x03\x12\x0f\n\x07members\x18\x04 \x03(\x04\",\n\tVenueList\x12\x1f\n\x06venues\x18\x01 \x03(\x0b\x32\x0f.synthpop.Venue\"e\n\x0bInfoPerMSOA\x12\x1e\n\x05shape\x18\x01 \x03(\x0b\x32\x0f.synthpop.Point\x12\x12\n\npopulation\x18\x02 \x01(\x04\x12\"\n\tbuildings\x18\x03 \x03(\x0b\x32\x0f.synthpop.Point\",\n\x05Point\x12\x11\n\tlongitude\x18\x01 \x01(\x02\x12\x10\n\x08latitude\x18\x02 \x01(\x02\"\xcf\x01\n\x06Person\x12\n\n\x02id\x18\x01 \x01(\x04\x12\x11\n\thousehold\x18\x02 \x01(\x04\x12!\n\x08location\x18\x03 \x01(\x0b\x32\x0f.synthpop.Point\x12\x10\n\x08orig_pid\x18\x04 \x01(\x03\x12\x0f\n\x07sic1d07\x18\x05 \x01(\x04\x12\x11\n\tage_years\x18\x06 \x01(\r\x12 \n\x06health\x18\x07 \x01(\x0b\x32\x10.synthpop.Health\x12+\n\x12\x66lows_per_activity\x18\x08 \x03(\x0b\x32\x0f.synthpop.Flows\"v\n\x06Health\x12\"\n\x07obesity\x18\x01 \x01(\x0e\x32\x11.synthpop.Obesity\x12\x1e\n\x16\x63\x61rdiovascular_disease\x18\x02 \x01(\r\x12\x10\n\x08\x64iabetes\x18\x03 \x01(\r\x12\x16\n\x0e\x62lood_pressure\x18\x04 \x01(\r\"g\n\x05\x46lows\x12$\n\x08\x61\x63tivity\x18\x01 \x01(\x0e\x32\x12.synthpop.Activity\x12\x1d\n\x05\x66lows\x18\x02 \x03(\x0b\x32\x0e.synthpop.Flow\x12\x19\n\x11\x61\x63tivity_duration\x18\x03 \x01(\x01\"(\n\x04\x46low\x12\x10\n\x08venue_id\x18\x01 \x01(\x04\x12\x0e\n\x06weight\x18\x02 \x01(\x01\"i\n\x05Venue\x12\n\n\x02id\x18\x01 \x01(\x04\x12$\n\x08\x61\x63tivity\x18\x02 \x01(\x0e\x32\x12.synthpop.Activity\x12!\n\x08location\x18\x03 \x01(\x0b\x32\x0f.synthpop.Point\x12\x0b\n\x03urn\x18\x04 \x01(\x04*c\n\x08\x41\x63tivity\x12\n\n\x06RETAIL\x10\x00\x12\x12\n\x0ePRIMARY_SCHOOL\x10\x01\x12\x14\n\x10SECONDARY_SCHOOL\x10\x02\x12\x08\n\x04HOME\x10\x03\x12\x08\n\x04WORK\x10\x04\x12\r\n\tNIGHTCLUB\x10\x05*L\n\x07Obesity\x12\x0b\n\x07OBESE_3\x10\x00\x12\x0b\n\x07OBESE_2\x10\x01\x12\x0b\n\x07OBESE_1\x10\x02\x12\x0e\n\nOVERWEIGHT\x10\x03\x12\n\n\x06NORMAL\x10\x04\x62\x06proto3')
)

_ACTIVITY = _descriptor.EnumDescriptor(
  name='Activity',
  full_name='synthpop.Activity',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='RETAIL', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PRIMARY_SCHOOL', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SECONDARY_SCHOOL', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HOME', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='WORK', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NIGHTCLUB', index=5, number=5,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1261,
  serialized_end=1360,
)
_sym_db.RegisterEnumDescriptor(_ACTIVITY)

Activity = enum_type_wrapper.EnumTypeWrapper(_ACTIVITY)
_OBESITY = _descriptor.EnumDescriptor(
  name='Obesity',
  full_name='synthpop.Obesity',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='OBESE_3', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OBESE_2', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OBESE_1', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OVERWEIGHT', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NORMAL', index=4, number=4,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1362,
  serialized_end=1438,
)
_sym_db.RegisterEnumDescriptor(_OBESITY)

Obesity = enum_type_wrapper.EnumTypeWrapper(_OBESITY)
RETAIL = 0
PRIMARY_SCHOOL = 1
SECONDARY_SCHOOL = 2
HOME = 3
WORK = 4
NIGHTCLUB = 5
OBESE_3 = 0
OBESE_2 = 1
OBESE_1 = 2
OVERWEIGHT = 3
NORMAL = 4



_POPULATION_VENUESPERACTIVITYENTRY = _descriptor.Descriptor(
  name='VenuesPerActivityEntry',
  full_name='synthpop.Population.VenuesPerActivityEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='synthpop.Population.VenuesPerActivityEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='synthpop.Population.VenuesPerActivityEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=254,
  serialized_end=331,
)

_POPULATION_INFOPERMSOAENTRY = _descriptor.Descriptor(
  name='InfoPerMsoaEntry',
  full_name='synthpop.Population.InfoPerMsoaEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='synthpop.Population.InfoPerMsoaEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='synthpop.Population.InfoPerMsoaEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=333,
  serialized_end=406,
)

_POPULATION = _descriptor.Descriptor(
  name='Population',
  full_name='synthpop.Population',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='households', full_name='synthpop.Population.households', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='people', full_name='synthpop.Population.people', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='venues_per_activity', full_name='synthpop.Population.venues_per_activity', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='info_per_msoa', full_name='synthpop.Population.info_per_msoa', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_POPULATION_VENUESPERACTIVITYENTRY, _POPULATION_INFOPERMSOAENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=29,
  serialized_end=406,
)


_HOUSEHOLD = _descriptor.Descriptor(
  name='Household',
  full_name='synthpop.Household',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='synthpop.Household.id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='msoa', full_name='synthpop.Household.msoa', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='orig_hid', full_name='synthpop.Household.orig_hid', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='members', full_name='synthpop.Household.members', index=3,
      number=4, type=4, cpp_type=4, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=408,
  serialized_end=480,
)


_VENUELIST = _descriptor.Descriptor(
  name='VenueList',
  full_name='synthpop.VenueList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='venues', full_name='synthpop.VenueList.venues', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=482,
  serialized_end=526,
)


_INFOPERMSOA = _descriptor.Descriptor(
  name='InfoPerMSOA',
  full_name='synthpop.InfoPerMSOA',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='shape', full_name='synthpop.InfoPerMSOA.shape', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='population', full_name='synthpop.InfoPerMSOA.population', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='buildings', full_name='synthpop.InfoPerMSOA.buildings', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=528,
  serialized_end=629,
)


_POINT = _descriptor.Descriptor(
  name='Point',
  full_name='synthpop.Point',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='longitude', full_name='synthpop.Point.longitude', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='latitude', full_name='synthpop.Point.latitude', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=631,
  serialized_end=675,
)


_PERSON = _descriptor.Descriptor(
  name='Person',
  full_name='synthpop.Person',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='synthpop.Person.id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='household', full_name='synthpop.Person.household', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='location', full_name='synthpop.Person.location', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='orig_pid', full_name='synthpop.Person.orig_pid', index=3,
      number=4, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sic1d07', full_name='synthpop.Person.sic1d07', index=4,
      number=5, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='age_years', full_name='synthpop.Person.age_years', index=5,
      number=6, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='health', full_name='synthpop.Person.health', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='flows_per_activity', full_name='synthpop.Person.flows_per_activity', index=7,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=678,
  serialized_end=885,
)


_HEALTH = _descriptor.Descriptor(
  name='Health',
  full_name='synthpop.Health',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='obesity', full_name='synthpop.Health.obesity', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cardiovascular_disease', full_name='synthpop.Health.cardiovascular_disease', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='diabetes', full_name='synthpop.Health.diabetes', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='blood_pressure', full_name='synthpop.Health.blood_pressure', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=887,
  serialized_end=1005,
)


_FLOWS = _descriptor.Descriptor(
  name='Flows',
  full_name='synthpop.Flows',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='activity', full_name='synthpop.Flows.activity', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='flows', full_name='synthpop.Flows.flows', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='activity_duration', full_name='synthpop.Flows.activity_duration', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1007,
  serialized_end=1110,
)


_FLOW = _descriptor.Descriptor(
  name='Flow',
  full_name='synthpop.Flow',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='venue_id', full_name='synthpop.Flow.venue_id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight', full_name='synthpop.Flow.weight', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1112,
  serialized_end=1152,
)


_VENUE = _descriptor.Descriptor(
  name='Venue',
  full_name='synthpop.Venue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='synthpop.Venue.id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='activity', full_name='synthpop.Venue.activity', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='location', full_name='synthpop.Venue.location', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='urn', full_name='synthpop.Venue.urn', index=3,
      number=4, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1154,
  serialized_end=1259,
)

_POPULATION_VENUESPERACTIVITYENTRY.fields_by_name['value'].message_type = _VENUELIST
_POPULATION_VENUESPERACTIVITYENTRY.containing_type = _POPULATION
_POPULATION_INFOPERMSOAENTRY.fields_by_name['value'].message_type = _INFOPERMSOA
_POPULATION_INFOPERMSOAENTRY.containing_type = _POPULATION
_POPULATION.fields_by_name['households'].message_type = _HOUSEHOLD
_POPULATION.fields_by_name['people'].message_type = _PERSON
_POPULATION.fields_by_name['venues_per_activity'].message_type = _POPULATION_VENUESPERACTIVITYENTRY
_POPULATION.fields_by_name['info_per_msoa'].message_type = _POPULATION_INFOPERMSOAENTRY
_VENUELIST.fields_by_name['venues'].message_type = _VENUE
_INFOPERMSOA.fields_by_name['shape'].message_type = _POINT
_INFOPERMSOA.fields_by_name['buildings'].message_type = _POINT
_PERSON.fields_by_name['location'].message_type = _POINT
_PERSON.fields_by_name['health'].message_type = _HEALTH
_PERSON.fields_by_name['flows_per_activity'].message_type = _FLOWS
_HEALTH.fields_by_name['obesity'].enum_type = _OBESITY
_FLOWS.fields_by_name['activity'].enum_type = _ACTIVITY
_FLOWS.fields_by_name['flows'].message_type = _FLOW
_VENUE.fields_by_name['activity'].enum_type = _ACTIVITY
_VENUE.fields_by_name['location'].message_type = _POINT
DESCRIPTOR.message_types_by_name['Population'] = _POPULATION
DESCRIPTOR.message_types_by_name['Household'] = _HOUSEHOLD
DESCRIPTOR.message_types_by_name['VenueList'] = _VENUELIST
DESCRIPTOR.message_types_by_name['InfoPerMSOA'] = _INFOPERMSOA
DESCRIPTOR.message_types_by_name['Point'] = _POINT
DESCRIPTOR.message_types_by_name['Person'] = _PERSON
DESCRIPTOR.message_types_by_name['Health'] = _HEALTH
DESCRIPTOR.message_types_by_name['Flows'] = _FLOWS
DESCRIPTOR.message_types_by_name['Flow'] = _FLOW
DESCRIPTOR.message_types_by_name['Venue'] = _VENUE
DESCRIPTOR.enum_types_by_name['Activity'] = _ACTIVITY
DESCRIPTOR.enum_types_by_name['Obesity'] = _OBESITY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Population = _reflection.GeneratedProtocolMessageType('Population', (_message.Message,), dict(

  VenuesPerActivityEntry = _reflection.GeneratedProtocolMessageType('VenuesPerActivityEntry', (_message.Message,), dict(
    DESCRIPTOR = _POPULATION_VENUESPERACTIVITYENTRY,
    __module__ = 'synthpop_pb2'
    # @@protoc_insertion_point(class_scope:synthpop.Population.VenuesPerActivityEntry)
    ))
  ,

  InfoPerMsoaEntry = _reflection.GeneratedProtocolMessageType('InfoPerMsoaEntry', (_message.Message,), dict(
    DESCRIPTOR = _POPULATION_INFOPERMSOAENTRY,
    __module__ = 'synthpop_pb2'
    # @@protoc_insertion_point(class_scope:synthpop.Population.InfoPerMsoaEntry)
    ))
  ,
  DESCRIPTOR = _POPULATION,
  __module__ = 'synthpop_pb2'
  # @@protoc_insertion_point(class_scope:synthpop.Population)
  ))
_sym_db.RegisterMessage(Population)
_sym_db.RegisterMessage(Population.VenuesPerActivityEntry)
_sym_db.RegisterMessage(Population.InfoPerMsoaEntry)

Household = _reflection.GeneratedProtocolMessageType('Household', (_message.Message,), dict(
  DESCRIPTOR = _HOUSEHOLD,
  __module__ = 'synthpop_pb2'
  # @@protoc_insertion_point(class_scope:synthpop.Household)
  ))
_sym_db.RegisterMessage(Household)

VenueList = _reflection.GeneratedProtocolMessageType('VenueList', (_message.Message,), dict(
  DESCRIPTOR = _VENUELIST,
  __module__ = 'synthpop_pb2'
  # @@protoc_insertion_point(class_scope:synthpop.VenueList)
  ))
_sym_db.RegisterMessage(VenueList)

InfoPerMSOA = _reflection.GeneratedProtocolMessageType('InfoPerMSOA', (_message.Message,), dict(
  DESCRIPTOR = _INFOPERMSOA,
  __module__ = 'synthpop_pb2'
  # @@protoc_insertion_point(class_scope:synthpop.InfoPerMSOA)
  ))
_sym_db.RegisterMessage(InfoPerMSOA)

Point = _reflection.GeneratedProtocolMessageType('Point', (_message.Message,), dict(
  DESCRIPTOR = _POINT,
  __module__ = 'synthpop_pb2'
  # @@protoc_insertion_point(class_scope:synthpop.Point)
  ))
_sym_db.RegisterMessage(Point)

Person = _reflection.GeneratedProtocolMessageType('Person', (_message.Message,), dict(
  DESCRIPTOR = _PERSON,
  __module__ = 'synthpop_pb2'
  # @@protoc_insertion_point(class_scope:synthpop.Person)
  ))
_sym_db.RegisterMessage(Person)

Health = _reflection.GeneratedProtocolMessageType('Health', (_message.Message,), dict(
  DESCRIPTOR = _HEALTH,
  __module__ = 'synthpop_pb2'
  # @@protoc_insertion_point(class_scope:synthpop.Health)
  ))
_sym_db.RegisterMessage(Health)

Flows = _reflection.GeneratedProtocolMessageType('Flows', (_message.Message,), dict(
  DESCRIPTOR = _FLOWS,
  __module__ = 'synthpop_pb2'
  # @@protoc_insertion_point(class_scope:synthpop.Flows)
  ))
_sym_db.RegisterMessage(Flows)

Flow = _reflection.GeneratedProtocolMessageType('Flow', (_message.Message,), dict(
  DESCRIPTOR = _FLOW,
  __module__ = 'synthpop_pb2'
  # @@protoc_insertion_point(class_scope:synthpop.Flow)
  ))
_sym_db.RegisterMessage(Flow)

Venue = _reflection.GeneratedProtocolMessageType('Venue', (_message.Message,), dict(
  DESCRIPTOR = _VENUE,
  __module__ = 'synthpop_pb2'
  # @@protoc_insertion_point(class_scope:synthpop.Venue)
  ))
_sym_db.RegisterMessage(Venue)


_POPULATION_VENUESPERACTIVITYENTRY._options = None
_POPULATION_INFOPERMSOAENTRY._options = None
# @@protoc_insertion_point(module_scope)