set -e
set -x

export PYTHONPATH="$PWD"
coverage run -m --source=uncertainty_wizard unittest discover tests_unit
coverage report -m
coverage xml
