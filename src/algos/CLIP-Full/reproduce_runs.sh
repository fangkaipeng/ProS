# DomainNet
python3 -u main.py > CLIP-Full_ip.log -debug_mode 0 -hd infograph -sd painting -bs 256 -es 5 -lr 0.000001 -clip_backbone ViT-B/32 -log 200 -e 100 -udcdr 0
python3 -u main.py > CLIP-Full_sq.log -debug_mode 0 -hd sketch -sd quickdraw -bs 256 -es 2 -lr 0.000001 -clip_backbone ViT-B/32 -log 200 -e 100 -udcdr 0
python3 -u main.py > CLIP-Full_cp.log -debug_mode 0 -hd clipart -sd painting -bs 256 -es 2 -lr 0.000001 -clip_backbone ViT-B/32 -log 200 -e 100 -udcdr 0
python3 -u main.py > CLIP-Full_pi.log -debug_mode 0 -hd painting -sd infograph -bs 256 -es 2 -lr 0.000001 -clip_backbone ViT-B/32 -log 200 -e 100 -ts FT -udcdr 0
python3 -u main.py > CLIP-Full_qs.log -debug_mode 0 -hd quickdraw -sd sketch -bs 256 -es 2 -lr 0.000001 -clip_backbone ViT-B/32 -log 200 -e 100 -udcdr 0

# Sketchy
python3 -u main.py > CLIP-Full_sketchy.log -data Sketchy -bs 480 -es 3 -lr 0.000001 -clip_backbone ViT-B/32 -log 15 -e 100 -debug_mode 0

# TUBerlin
python3 -u main.py > CLIP-Full_TUberlin.log -data TUBerlin -bs 480 -es 3 -lr 0.000001 -clip_backbone ViT-B/32 -log 15 -e 100 -debug_mode 0
