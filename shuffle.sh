#!/bin/bash

for var in $(seq 0 19); do
    python shuffle.py $var
done
