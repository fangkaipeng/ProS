python3 -u test.py > UCDR_sq.log  -hd sketch -sd quickdraw -bs 50 -log 15 -debug_mode 0 -resume INSERT_MODEL_PATH_HERE_LIKE_xx.pth
python3 -u test.py > UCDR_qs.log  -hd quickdraw -sd sketch -bs 50 -log 15 -debug_mode 0 -resume INSERT_MODEL_PATH_HERE_LIKE_xx.pth
python3 -u test.py > UCDR_pi.log  -hd painting -sd infograph -bs 50 -log 15 -debug_mode 0 -resume INSERT_MODEL_PATH_HERE_LIKE_xx.pth
python3 -u test.py > UCDR_cp.log  -hd clipart -sd painting -bs 50 -log 15 -debug_mode 0 -resume INSERT_MODEL_PATH_HERE_LIKE_xx.pth
python3 -u test.py > UCDR_ip.log  -hd infograph -sd painting -bs 50 -log 15 -debug_mode 0 -resume INSERT_MODEL_PATH_HERE_LIKE_xx.pth
