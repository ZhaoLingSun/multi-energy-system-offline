#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-intranet}
REMOTE_DIR=${REMOTE_DIR:-/home/ace/workspaces/python_workspace/multi-energy-offline}
IMAGE=${IMAGE:-meos-offline}
MIP_GAP=${MIP_GAP:-0.1}
DAYS=${DAYS:-7}
THREADS=${THREADS:-4}
TIME_LIMIT=${TIME_LIMIT:-300}

rsync -az \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'runs' \
  --exclude 'output' \
  ./ ${REMOTE_HOST}:${REMOTE_DIR}/

ssh ${REMOTE_HOST} "cd ${REMOTE_DIR} && docker build -t ${IMAGE} ."
ssh ${REMOTE_HOST} "cd ${REMOTE_DIR} && docker run --rm \
  -e GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \
  -v /opt/gurobi:/opt/gurobi \
  -v ${REMOTE_DIR}:/app \
  ${IMAGE} \
  python run_all.py --mip-gap ${MIP_GAP} --days ${DAYS} --threads ${THREADS} --time-limit ${TIME_LIMIT}"
