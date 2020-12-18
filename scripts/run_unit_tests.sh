set -e
set -x

export PYTHONPATH="$PWD"
python -m unittest discover tests_unit