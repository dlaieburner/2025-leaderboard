#!/bin/bash
while true; do
  echo "===== Running at $(date) ====="
  ../botograder/scripts/lb_canvas.py `cat CANVAS_ASSIGNMENT_URL.txt` lb_test
  echo "Sleeping 10 minutes..."
  sleep 600
done
