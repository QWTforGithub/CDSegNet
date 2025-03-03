#!/bin/bash
cd ../libs

cd pointgroup_ops
python setup.py install
echo "pointgroup_ops --> Finishing!"
cd ../

cd pointops
python setup.py install
echo "pointops --> Finishing!"
cd ../

cd pointops2
python setup.py install
echo "pointops2 --> Finishing!"
