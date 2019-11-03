rm log*.txt
nohup python make_prediction.py 0 >> log.txt &
nohup python make_prediction.py 1 -n >> log_n.txt &
nohup python make_prediction.py 2 -e >> log_e.txt &
nohup python make_prediction.py 3 -n -e >> log_en.txt &