#!/bin/bash
while true; do
  echo "===== Running at $(date) ====="
  ../botograder/scripts/lb_canvas.py `cat canvas_assignment_url.txt` lb_test
  echo "Sleeping 20 minutes..."
  sleep 1200
done
