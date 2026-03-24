rm -rf diboson_output_trigger/debug_test

python3 processor_diboson_trigger.py data/2017/config_diboson_trigger.hjson -o diboson_output_trigger/debug_test --statedata diboson_output_trigger/debug_test/state.coffea -d  


#python scripts/plot_histograms.py data/2017/config_diboson_trigger_NUM.hjson diboson_output_trigger_NUM/debug_test/hists/hists.json  

