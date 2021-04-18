#!/bin/bash

mkdir -p galaxy_images/train/Elliptical
mkdir -p galaxy_images/train/Non-Elliptical
mkdir -p galaxy_images/validation/Elliptical
mkdir -p galaxy_images/validation/Non-Elliptical

rm -rf galaxy_images/train/Elliptical/*
rm -rf galaxy_images/train/Non-Elliptical/*
rm -rf galaxy_images/validation/Elliptical/*
rm -rf galaxy_images/validation/Non-Elliptical/*

python3 gen-plots.py
python3 -m venv venv;
source venv/bin/activate;
python3 cnn-galaxy.py;
