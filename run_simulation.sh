#!/usr/bin/env bash

./clean_output.sh
mpiexec -np $1 python3 rcc.py
