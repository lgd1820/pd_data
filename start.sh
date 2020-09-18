#!/bin/bash

for var in $(seq 0 19); do
    python model.py $var
done
