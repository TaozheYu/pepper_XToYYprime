source /cvmfs/cms.cern.ch/cmsset_default.sh
source /cvmfs/grid.desy.de/etc/profile.d/grid-ui-env.sh > /dev/null
source /cvmfs/cms.cern.ch/rucio/setup-py3.sh
voms-proxy-init --rfc --voms cms --out ~/.globus/x509up
export X509_USER_PROXY=~/.globus/x509up

