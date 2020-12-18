#!/bin/bash

DOCKER_DIR="./docker/"

# Parse flags

FOR_GPU='false'
FULL_BUILD='false'
TAG=""

# Process all options supplied on the command line
while getopts ":t:gf" opt; do
  case $opt in
  'g')
    # Update the value of the option x flag we defined above
    FOR_GPU='true'
    echo "Building image for GPU"
    ;;
  'f')
    # Update the value of the option x flag we defined above
    FULL_BUILD='true'
    echo "Building full image (i.e., including source and test files)"
    ;;
  't')
    TAG=$OPTARG
    echo "Building with tag '$TAG'"
    ;;
  '?')
    echo "INVALID OPTION - ${OPTARG}" >&2
    exit 1
    ;;
  ':')
    echo "MISSING ARGUMENT for option - ${OPTARG}" >&2
    exit 1
    ;;
  *)
    echo "UNIMPLEMENTED OPTION - ${OPTKEY}" >&2
    exit 1
    ;;
  esac
done

shift $((OPTIND - 1))

PROJECT_NAME="${1}"

# Make sure we run in root folder of repository
if ! [[ -d "$PROJECT_NAME" ]]; then
  printf "ERROR: Folder %s does not exist. " "${PROJECT_NAME}"
  printf "Make sure to run this script from the root folder of your repo and that you pass your "
  printf "project name (source folder name) as first argument to this script. "
  echo "(e.g., ./scripts/build_docker_image.sh -g -f my_project)"
  exit 1
fi


if [[ "" == "$TAG" ]]
then
  TAG="${PROJECT_NAME}/${PROJECT_NAME}:snapshot"
  echo "Using default tag: '$TAG'"
fi

if [[ 'true' == "$FOR_GPU" ]]
then
  PU_FOLDER="${DOCKER_DIR}gpu/"
else
  PU_FOLDER="${DOCKER_DIR}cpu/"
fi

if [[ 'true' == "$FULL_BUILD" ]]
then
  FULL_FOLDER="${PU_FOLDER}full/"
else
  FULL_FOLDER="${PU_FOLDER}env/"
fi

DOCKERFILE="${FULL_FOLDER}Dockerfile"

docker build -f "$DOCKERFILE" -t "$TAG" .