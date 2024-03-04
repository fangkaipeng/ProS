import argparse
import numpy as np
import os
import pickle

root_path = '/media/mvp18/Study/ZS-SBIR/datasets/TU-Berlin/'

word_replace_dict = {'alarm_clock':'clock', 'car_(sedan)':'sedan', 'sea_turtle':'turtle', 'teddy_bear':'teddybears', 'wine_bottle':'bottle', 
					 'pickup_truck':'truck', 'hot_dog':'hotdog', 'paper_clip':'paperclip', 'cell_phone':'cell-phone', 'bear_(animal)':'bear', 
					 'santa_claus':'santa', 'beer_mug':'mug', 'ice_cream_cone':'ice-cream', 'crane_(machine)':'crane','race_car':'race-car', 
					 'head_phones':'headphones', 'speed_boat':'speedboat', 'sponge_bob':'spongebob', 'wrist_watch':'wristwatch', 
					 'tennis_racket':'racket', 'human_skeleton':'skeleton', 'power_outlet':'switchboard', 'walkie_talkie':'walkie-talkie'
					}

all_classes = sorted(os.listdir(os.path.join(root_path, 'images')))

with open('test_classes_random_split.txt', 'r') as fp:
    te_classes = fp.read().splitlines()

with open('val_classes.txt', 'r') as fp:
    va_classes = fp.read().splitlines()

tr_classes = np.setdiff1d(all_classes, np.union1d(te_classes, va_classes))

# glove_dict_path = '/media/mvp18/Study/SBIR/datasets/glove.6B/glove.6B.200d.txt'
glove_dict_path = '/media/mvp18/Study/ZS-SBIR/datasets/glove.6B/glove.6B.300d.txt'
embeddings_dict = {}
with open(glove_dict_path, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        token = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[token] = vector

clss_embed = {}
i=0
for clss in tr_classes:
	if clss in word_replace_dict:
		dict_ele = word_replace_dict[clss]
		clss_embed[clss] = embeddings_dict[dict_ele]
	else:
		clss_embed[clss] = embeddings_dict[clss]
	# if clss not in embeddings_dict.keys() and clss not in word_replace_dict:
	# 	i+=1
	# 	print(clss)

print(clss_embed.keys())
print(len(clss_embed))

with open('glove300.pkl', 'wb') as f:
	pickle.dump(clss_embed, f)