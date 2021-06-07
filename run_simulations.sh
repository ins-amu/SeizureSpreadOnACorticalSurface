#!/bin/bash

python simulate.py simConfig/Real.json
python simulate.py simConfig/S-Realistic.json
python simulate.py simConfig/S-Flat.json
python simulate.py simConfig/S-Sine.json

