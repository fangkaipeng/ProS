# UCDR protocol for DomainNet
python3 test.py > CLIP-Full_ip.log -hd infograph -sd painting -debug_mode 0 -bs 256 -es 5 -clip_backbone ViT-B/32 -log 200 -resume INSERT_MODEL_NAME_HERE_LIKE_xx.pth
python3 test.py > CLIP-Full_pi.log -hd painting -sd infograph -debug_mode 0 -bs 256 -es 5 -clip_backbone ViT-B/32 -log 200 -resume INSERT_MODEL_NAME_HERE_LIKE_xx.pth
python3 test.py > CLIP-Full_sq.log -hd sketch -sd quickdraw   -debug_mode 0 -bs 256 -es 5 -clip_backbone ViT-B/32 -log 200 -resume INSERT_MODEL_NAME_HERE_LIKE_xx.pth
python3 test.py > CLIP-Full_qs.log -hd quickdraw -sd sketch   -debug_mode 0 -bs 256 -es 5 -clip_backbone ViT-B/32 -log 200 -resume INSERT_MODEL_NAME_HERE_LIKE_xx.pth
python3 test.py > CLIP-Full_cp.log -hd clipart -sd painting   -debug_mode 0 -bs 256 -es 5 -clip_backbone ViT-B/32 -log 200 -resume INSERT_MODEL_NAME_HERE_LIKE_xx.pth


# UdCDR protocol for DomainNet
python3 test.py > CLIP-Full_ip.log -udcdr 1 -hd infograph -sd painting -debug_mode 0 -bs 256 -es 5 -clip_backbone ViT-B/32 -log 200 -resume INSERT_MODEL_NAME_HERE_LIKE_xx.pth
python3 test.py > CLIP-Full_pi.log -udcdr 1 -hd painting -sd infograph -debug_mode 0 -bs 256 -es 5 -clip_backbone ViT-B/32 -log 200 -resume INSERT_MODEL_NAME_HERE_LIKE_xx.pth
python3 test.py > CLIP-Full_sq.log -udcdr 1 -hd sketch -sd quickdraw   -debug_mode 0 -bs 256 -es 5 -clip_backbone ViT-B/32 -log 200 -resume INSERT_MODEL_NAME_HERE_LIKE_xx.pth
python3 test.py > CLIP-Full_qs.log -udcdr 1 -hd quickdraw -sd sketch   -debug_mode 0 -bs 256 -es 5 -clip_backbone ViT-B/32 -log 200 -resume INSERT_MODEL_NAME_HERE_LIKE_xx.pth
python3 test.py > CLIP-Full_cp.log -udcdr 1 -hd clipart -sd painting   -debug_mode 0 -bs 256 -es 5 -clip_backbone ViT-B/32 -log 200 -resume INSERT_MODEL_NAME_HERE_LIKE_xx.pth


# UCDR protocol for Sketchy
python3 test.py > CLIP-Full_sketchy.log -data Sketchy -bs 480 -es 3 -clip_backbone ViT-B/32 -log 15 -debug_mode 0 -resume INSERT_MODEL_NAME_HERE_LIKE_xx.pth

# UdCDR protocol for TU-Berlin
python3 test.py > CLIP-Full_TUBerlin.log -data TUBerlin -bs 480 -es 3 -clip_backbone ViT-B/32 -log 15 -debug_mode 0 -resume INSERT_MODEL_NAME_HERE_LIKE_xx.pth