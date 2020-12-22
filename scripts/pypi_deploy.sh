#! /bin/sh


# Process all options supplied on the command line
while getopts "p" opt; do
  case $opt in
  'p')
    # Update the value of the option x flag we defined above
    PROD=true
    echo "Deploying to production pypi"
    ;;
  *)
    echo "UNIMPLEMENTED OPTION - ${OPTKEY}" >&2
    exit 1
    ;;
  esac
done

rm dist/*
python3 setup.py sdist bdist_wheel

# Make sure we run in root folder of repository
if ! [ "$PROD" = true  ]; then
  echo "Deploying to TEST (test.pypi). Use the -p flag to deploy to prod."
  twine upload --repository testpypi --skip-existing dist/*
else
  twine upload --skip-existing dist/*
fi
