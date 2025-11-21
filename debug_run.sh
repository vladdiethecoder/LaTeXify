#!/bin/bash
export PATH=$PWD/bin:$PATH
python3 -u run_latexify.py > debug_run.log 2>&1
echo "Exit Code: $?" >> debug_run.log
