#!/bin/bash

cd ../libs

cd pointgroup_ops
python install setup.py
echo "---- pointgroup_ops--->Finish! ----"

cd ../pointops
python setup.py install
echo "---- pointops--->Finish! ----"

cd ../pointops2
python setup.py install
echo "---- pointops2--->Finish! ----"
