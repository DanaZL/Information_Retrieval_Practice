
# coding: utf-8:

import sys
import re
import random
from operator import itemgetter
import urlparse
import urllib

def insert_features(key, features):
    if features.get(key):   #4a
        features[key] += 1
    else:
        features[key] = 1

def add_features(path, query, features):
    cnt_path_segments = 0
    
    path = urllib.unquote(path)
    path_segment = path.split('/')
    
    for segm in path_segment:
        if segm == "":
            continue
        
        insert_features("segment_name_" + str(cnt_path_segments) + ":" + segm, features) #4a
        
        #if re.match(r"[0-9]+|[0-9]+\.", segm):
        if re.match(r"[0-9]+$|[0-9]+\.", segm):
            insert_features("segment_[0-9]_" + str(cnt_path_segments) + ":1", features)   #4b
                
        if re.match(r"[^\d]+\d+[^\d]+$", segm):
           insert_features("segment_substr[0-9]_" + str(cnt_path_segments) + ":1", features)   #4c

        match = re.match(r".+\.(.+)", segm)
        if match:
            insert_features("segment_ext_" + str(cnt_path_segments) + ":" + match.group(1), features)   #4d
            
        match = re.match(r"[^\d]+\d+[^\d]+\.(\W+)", segm)
        if match:
            insert_features("segment_ext_substr[0-9]_" + str(cnt_path_segments) + ":" + match.group(1), features)   #4e
        
        insert_features("segment_len_" + str(cnt_path_segments) + ":" + str(len(segm)), features)  #4f
        
        cnt_path_segments += 1
    
    insert_features("segments:" + str(cnt_path_segments), features)  #4f
    
    query = urllib.unquote(query)
    query_segment = query.split('&')
    
    if (query == ""):
                    """if features.get("param_name:"):
                        features["param_name:"] += 1
                    else:
                        features["param_name:"] = 1"""
    else:
        for segm in query_segment:
            if segm == "":
                continue
        
            match = re.match(r"(.+)=(.+)", segm)
            if match:
                insert_features("param_name:" + match.group(1), features)
                insert_features("param:" + match.group(), features)    
            else:
                insert_features("param_name:" + segm, features)
        

def extract_features(INPUT_FILE_1, INPUT_FILE_2, OUTPUT_FILE):
    features = {}  #name_features->cnt_festures
    cnt_examined_sample = 1000
    alpha = 0.05
    
    with open(INPUT_FILE_1, "r") as examined_file:
        examined_lines = []
        for line in examined_file:
            examined_lines.append(line[:-1])
    examined_file.close()
    
    examined_sample = range(len(examined_lines))
    random.shuffle(examined_sample)
    examined_sample = examined_sample[:cnt_examined_sample]
    
    with open(INPUT_FILE_2, "r") as general_file:
        general_lines = []
        for line in general_file:
            general_lines.append(line[:-1])
    general_file.close()
    
    general_sample = range(len(general_lines))
    random.shuffle(general_sample)
    general_sample = general_sample[:cnt_examined_sample]
    
    for i in examined_sample:
        parsed = urlparse.urlparse(examined_lines[i])
        add_features(parsed.path, parsed.query, features)
        
    for i in general_sample:
        parsed = urlparse.urlparse(general_lines[i])
        add_features(parsed.path, parsed.query, features)
    
    out = open(OUTPUT_FILE, "w")
    border = 2 * alpha * cnt_examined_sample
    #features_main = {key: features[key] for key in features.keys() if features[key] > border} 
    for key in features.keys():
        if features[key] > border:
            #print key + '\t' + str(features[key])
            out.write(key + '\t' + str(features[key]) + '\n')

    out.close()