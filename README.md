This pepper framework was created to do XToYYprime analysis. Using this framework, you can produce the 3D histogram.

You can follow these steps to procude the histogram:

# Creat the local working area

Copy the project in the location

```bash
cd <YOUR_PATH>
git clone git@github.com:TaozheYu/pepper_XToYYprime.git
```

# Set the environment
Before doing this step, you should first setup your grid certificate, you can see here:
https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookStartingGrid#BasicGrid

```bash
source init.sh
source example/environment.sh 
```

# Make the config files
The config files are in the `data/<year>` directory, the name is something like `config_diboson_make3DTemplate.hjson`. In this config file, you should set these things:
1. the input dataset and MC sample(include signal and background)
2. the cross section file, it is `data/crosssections_13.hjson`, in this file you should set the corresponding cross section of input samples.
3. the luminosity factor file, it is `data/<year>/lumifactors_*.json`, you can run `python3 scripts/compute_mc_lumifactors.py data/2017/config_diboson_make3DTemplate.hjson data/2017/lumifactors.json` to produce this file

# Produce the 3D histogram
`processor_diboson_make3DTemplate.py` is used to run the resolved case,`processor_diboson_boosted_make3DTemplate.py` is used to run the boosted case. You can follow these command to run the code at debug model 

```bash
python3 processor_diboson_make3DTemplate.py data/2017/config_diboson_make3DTemplate.hjson -o diboson_output_make3DTemplate/debug_test --statedata diboson_output_make3DTemplate/debug_test/state.coffea -d
```

If you want to run all events, I suggest you submit the jobs to condor, the command is like this:

```bash
nohup python3 processor_diboson_make3DTemplate.py /afs/desy.de/user/t/tayu/private/pepper_exercise/data/2017/config_diboson_make3DTemplate.hjson -o /data/dust/user/tayu/diboson_output_make3DTemplate/debug_test --statedata /data/dust/user/tayu/diboson_output_make3DTemplate/debug_test/state.coffea -i /afs/desy.de/user/t/tayu/private/pepper_exercise/example/environment.sh -R -c 100 --condorlogdir /data/dust/user/tayu/pepper_logs > log.txt&
```

