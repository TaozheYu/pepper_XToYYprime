#!/bin/bash

echo "Starting job on " `date` #Date/time of start of job
echo "Runing on: `uname -a`" #Condor job is running on this node
echo "System software: `cat /etc/redhat-release`" #Operating System on that node

#source /cvmfs/cms.cern.ch/cmsset_default.sh
#export SCRAM_ARCH=slc7_amd64_gcc700
cd /afs/desy.de/user/t/tayu/private/pepper_exercise
source example/environment.sh 

python3 processor_diboson.py /afs/desy.de/user/t/tayu/private/pepper_exercise/data/2017/config_diboson.hjson -o /nfs/dust/cms/user/tayu/diboson_output/debug_test --statedata /nfs/dust/cms/user/tayu/diboson_output/debug_test/state.coffea  
#eval `scram runtime -sh`

