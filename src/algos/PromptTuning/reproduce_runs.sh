# CoOp
python3 -u main.py > Sketchy_TP.log -debug_mode 0 -data Sketchy -hd sketch -sd quickdraw  -bs 256  -es 2 -lr 0.001  -log 200 -e 100 -ts TP 
python3 -u main.py > TUBerlin_TP.log -debug_mode 0 -data TUBerlin -hd quickdraw -sd sketch  -bs 256  -es 2 -lr 0.001  -log 200 -e 100 -ts TP 
python3 -u main.py > pi_TP.log -debug_mode 0 -data DomainNet -hd painting -sd infograph -bs 256 -es 2 -lr 0.001  -log 200 -e 100 -ts TP
python3 -u main.py > ip_TP.log -debug_mode 0 -data DomainNet -hd infograph -sd painting -bs 256 -es 2 -lr 0.001  -log 200 -e 100 -ts TP
python3 -u main.py > cp_TP.log -debug_mode 0 -data DomainNet -hd clipart -sd painting  -bs 256  -es 2 -lr 0.001  -log 200 -e 100 -ts TP
python3 -u main.py > sq_TP.log -debug_mode 0 -data DomainNet -hd sketch -sd quickdraw  -bs 256  -es 2 -lr 0.001  -log 200 -e 100 -ts TP 
python3 -u main.py > qs_TP.log -debug_mode 0 -data DomainNet -hd quickdraw -sd sketch  -bs 256  -es 2 -lr 0.001  -log 200 -e 100 -ts TP

# VPT
python3 -u main.py > Sketchy_VP.log -debug_mode 0 -data Sketchy -hd sketch -sd quickdraw  -bs 256  -es 2 -lr 0.001  -log 200 -e 100 -ts VP 
python3 -u main.py > TUBerlin_VP.log -debug_mode 0 -data TUBerlin -hd quickdraw -sd sketch  -bs 256  -es 2 -lr 0.001  -log 200 -e 100 -ts VP 
python3 -u main.py > pi_VP.log -debug_mode 0 -data DomainNet -hd painting -sd infograph -bs 256 -es 2 -lr 0.001  -log 200 -e 100 -ts VP
python3 -u main.py > ip_VP.log -debug_mode 0 -data DomainNet -hd infograph -sd painting -bs 256 -es 2 -lr 0.001  -log 200 -e 100 -ts VP
python3 -u main.py > cp_VP.log -debug_mode 0 -data DomainNet -hd clipart -sd painting  -bs 256  -es 2 -lr 0.001  -log 200 -e 100 -ts VP
python3 -u main.py > sq_VP.log -debug_mode 0 -data DomainNet -hd sketch -sd quickdraw  -bs 256  -es 2 -lr 0.001  -log 200 -e 100 -ts VP 
python3 -u main.py > qs_VP.log -debug_mode 0 -data DomainNet -hd quickdraw -sd sketch  -bs 256  -es 2 -lr 0.001  -log 200 -e 100 -ts VP

