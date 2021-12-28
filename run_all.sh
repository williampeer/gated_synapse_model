screen -S `asbienv; nice -n 7 python run_exp_NLIF_model.py -mt NLIF -et AutoEncoding`
screen -S `asbienv; nice -n 7 python run_exp_NLIF_model.py -mt LIF -et AutoEncoding`
screen -S `asbienv; nice -n 7 python run_exp_NLIF_model.py -mt NLIF -et GeneralPredictiveEncoding`
screen -S `asbienv; nice -n 7 python run_exp_NLIF_model.py -mt LIF -et GeneralPredictiveEncoding`
