set -e
set -x

export PYTHONPATH="$PWD"
python -m docstr-coverage uncertainty_wizard --skipfiledoc --skip-private --failunder=100