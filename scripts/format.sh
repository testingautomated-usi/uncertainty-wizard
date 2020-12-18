#!/bin/sh -e
set -x

isort --profile black uncertainty_wizard tests_unit --profile black
autoflake --remove-all-unused-imports --recursive --remove-unused-variables \
  --in-place uncertainty_wizard tests_unit --exclude=__init__.py

black uncertainty_wizard tests_unit
isort --profile black uncertainty_wizard tests_unit --profile black
