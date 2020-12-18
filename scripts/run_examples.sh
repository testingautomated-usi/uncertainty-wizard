set -e
set -x

export PYTHONPATH="$PWD"

jupyter nbconvert --to script ./examples/*.ipynb --output-dir='./examples_build/'

for f in 'examples_build/'*.py; do python "$f"; done