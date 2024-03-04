import argparse
import numpy as np
import os
import pickle

root_path = '/media/mvp18/Study/ZS-SBIR/datasets/Sketchy/'

word_replace_dict = {'jack_o_lantern':'lantern', 'alarm_clock':'clock', 'car_(sedan)':'sedan', 'sea_turtle':'turtle', 'teddy_bear':'teddybears', 
					'wading_bird':'waders', 'wine_bottle':'bottle', 'hermit_crab':'crab', 'hot_air_balloon':'balloon', 'pickup_truck':'truck'}

classes = sorted(os.listdir(os.path.join(root_path, 'extended_photo')))

# glove_dict_path = '/media/mvp18/Study/SBIR/datasets/glove.6B/glove.6B.200d.txt'
glove_dict_path = '/media/mvp18/Study/SBIR/datasets/glove.6B/glove.6B.300d.txt'
embeddings_dict = {}
with open(glove_dict_path, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        token = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[token] = vector

clss_embed = {}
for clss in classes:
	if clss in word_replace_dict:
		dict_ele = word_replace_dict[clss]
		clss_embed[clss] = embeddings_dict[dict_ele]
	else:
		clss_embed[clss] = embeddings_dict[clss]
	# if clss not in embeddings_dict.keys():
	# 	print(clss)

print(clss_embed.keys())
print(len(clss_embed))

with open('Sketchy_glove300.pkl', 'wb') as f:
	pickle.dump(clss_embed, f)