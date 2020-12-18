#!/bin/bash

DOCKER_DIR="./docker/"
SOURCES_DIR="${1}"


if ! [[ -d "./$SOURCES_DIR" ]]; then
  printf "ERROR: Folder %s does not exist. " "${SOURCES_DIR}"
  printf "Make sure to run this script from the root folder of your repo and that you pass your "
  echo "project name (source folder name) as first argument to this script."
  exit 1
fi
if [[ "" == "$SOURCES_DIR"  ]]; then
  printf "ERROR: You did not specify your project name (source folder). "
  echo "Make sure to pass your project name (source folder name) as first positional argument to this script."
  exit 1
fi

# ===============================================
# Iterate and parse all lines of requirements.txt
# ===============================================
NON_TF_LINES=()

while IFS="" read -r line || [ -n "$line" ]; do

  # Check if line is tensorflow import with version specified. If so, read version number
  if [[ $line =~ tensorflow==* ]]; then
    TF_VERSION=$(echo $line | cut -d'=' -f 3)
    echo 'tensorflow dependency: ' "$TF_VERSION"

  # Check if line is tensorflow import without version specified or . If so, read version number
  elif [[ $line =~ "tensorflow>="* ]]; then
    echo 'tensorflow dependency: ' "latest"
    TF_VERSION="latest"

  # Check if tensorflow is in requirements.txt without version number
  elif [[ $line == 'tensorflow' ]]; then
    printf "WARNING: Please specify version for your tensorflow import (e.g., tensorflow==2.3.0)"
    echo "For now, we will just base the dockerfile on tensorflow/tensorflow:latest"
    TF_VERSION="latest"

  else
    if ! [[ $line = \#* || $line =~ ==* ]]
    then
      printf "WARNING: You did not specify an exact version of dependency '%s'. " "${line}"
      echo "Multiple builds of the generated dockerfile may lead to different images."
    fi
    NON_TF_LINES+=("$line")
  fi

done <requirements.txt

# ========================================================================
# Delete existing docker dir and re-generate folder structure from scratch
# ========================================================================
CPU_DIR="${DOCKER_DIR}cpu"
GPU_DIR="${DOCKER_DIR}gpu"
DEP_GPU_DIR="${GPU_DIR}/env/"
DEP_CPU_DIR="${CPU_DIR}/env/"
FULL_GPU_DIR="${GPU_DIR}/full/"
FULL_CPU_DIR="${CPU_DIR}/full/"
ALL_FOLDERS=("$DEP_GPU_DIR" "$DEP_CPU_DIR" "$FULL_GPU_DIR" "$FULL_CPU_DIR")
rm -rf "$DOCKER_DIR"
mkdir "$DOCKER_DIR"
mkdir "$CPU_DIR"
mkdir "$GPU_DIR"

for folder in "${ALL_FOLDERS[@]}"
do
  mkdir "$folder"
done

# ========================================================================
# Create the initial docker files with the tag specific FROM statements
# ========================================================================
for folder in "${ALL_FOLDERS[@]}"
  do
    {
      echo "# ======================================="
      echo "# This is an automatically generated file."
      echo "# ======================================="
    } >> "${folder}Dockerfile"
  done

echo "FROM tensorflow/tensorflow:${TF_VERSION}-gpu" >> "${DEP_GPU_DIR}Dockerfile"
echo "FROM tensorflow/tensorflow:${TF_VERSION}" >> "${DEP_CPU_DIR}Dockerfile"
echo "FROM tensorflow/tensorflow:${TF_VERSION}-gpu" >> "${FULL_GPU_DIR}Dockerfile"
echo "FROM tensorflow/tensorflow:${TF_VERSION}" >> "${FULL_CPU_DIR}Dockerfile"


# ==============================================================
# Create the requirements file without the tensorflow dependency
# ==============================================================

# Header
{
  echo "# ========================================================================================================="
  echo "# This is an automatically generated file. It has the same content as the root folder 'requirements.tex',"
  echo "# except for the tensorflow import (which is removed as it will be inherited from tensorflow docker image)"
  echo "# ========================================================================================================="
  echo ""
} >> "${DOCKER_DIR}requirements.txt"

# Print Lines
for req_line in "${NON_TF_LINES[@]}"
do
  echo "$req_line" >> "${DOCKER_DIR}requirements.txt"
done


# ==============================================================
# Pip install requirements file (this is the same for all tags)
# ==============================================================
for folder in "${ALL_FOLDERS[@]}"
  do
    {
      echo ""
      echo "# Update pip and install all pip dependencies"
      echo "RUN /usr/bin/python3 -m pip install --upgrade pip"
      echo "COPY ${DOCKER_DIR}requirements.txt /opt/project/requirements.txt"
      echo "RUN pip install -r /opt/project/requirements.txt"
    } >> "${folder}Dockerfile"
  done


# ===============
# COPY Resources
# ===============
FULL_FOLDERS=("$FULL_GPU_DIR" "$FULL_CPU_DIR")
for folder in "${FULL_FOLDERS[@]}"
  do
    {
      echo ""
      echo "# Copy the resources folder"
      echo "COPY ./resources /opt/project/resources"
    } >> "${folder}Dockerfile"
  done



# =====================================
# COPY Project Sources, including tests
# =====================================
for folder in "${FULL_FOLDERS[@]}"
  do
    {
      echo ""
      echo "# Copy full project (sources + tests). This does *NOT* include the mount folder."
      echo "COPY ./${SOURCES_DIR} /opt/project/${SOURCES_DIR}"
      echo "COPY ./tests /opt/project/tests"
    } >> "${folder}Dockerfile"
  done

echo "Regenerated ./docker/ folder based on your requirements.txt"