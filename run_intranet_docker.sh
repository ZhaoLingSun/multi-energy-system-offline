#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-intranet}
REMOTE_DIR=${REMOTE_DIR:-/home/ace/workspaces/python_workspace/multi-energy-offline}
IMAGE=${IMAGE:-meos-offline}
MIP_GAP=${MIP_GAP:-1e-3}
THREADS=${THREADS:-64}
TIME_LIMIT=${TIME_LIMIT:-}

rsync -az \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'runs' \
  --exclude 'output' \
  ./ ${REMOTE_HOST}:${REMOTE_DIR}/

ssh ${REMOTE_HOST} "cd ${REMOTE_DIR} && docker build -t ${IMAGE} ."

RUN_ARGS=(python run_all.py --mip-gap ${MIP_GAP} --threads ${THREADS})
if [ -n \"${TIME_LIMIT}\" ]; then
  RUN_ARGS+=(--time-limit \"${TIME_LIMIT}\")
fi

ssh ${REMOTE_HOST} "cd ${REMOTE_DIR} && docker run --rm \
  -e GRB_LICENSE_FILE=/app/gurobi.lic \
  -v /home/ace/gurobi.lic:/app/gurobi.lic:ro \
  -v ${REMOTE_DIR}:/app \
  ${IMAGE} \
  ${RUN_ARGS[*]}"
