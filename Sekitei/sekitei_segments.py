# coding: utf-8


import sys
import os
import re
import random
import time
from sklearn.cluster import KMeans
from sklearn.cluster import Birch

import numpy as np
import urlparse
import urllib
import operator


# sekitei = None# coding: utf-8:
class sekitei:
    pass


def insert_features(key, features, url_cnt):
    if not features.get(key):  # 4a
        features[key] = []
    features[key].append(url_cnt)


def add_features(path, query, features, url_cnt):
    cnt_path_segments = 0
    # #print path
    path = path.encode("ascii")

    if (path != urllib.unquote(path)) or  (query != urllib.unquote(query)):
        insert_features("url_was_encoded:1", features, url_cnt)
    path = urllib.unquote(path)
    if path[-1] == '/':
        insert_features("url_last_symbols_/:1", features,
                        url_cnt)  # 4b
    path_segment = path.split('/')

    match = re.match(r".+/(20\d\d)/", path)
    if match:
        if (int(match.group(1)) >= 2010):
            insert_features("url_data_large_2010_" + ":1", features,
                            url_cnt)  # 4b
        if re.match(r".+/\d\d/\d\d/?", path):
            insert_features("data_in_url_path:1", features, url_cnt)  # 4b

    cnt_digit_segments = 0
    for segm in path_segment:
        if segm == "":
            continue

        insert_features("segment_name_" + str(cnt_path_segments) + ":" + segm.lower(), features, url_cnt)  # 4a
        # if re.match(r"[0-9]+|[0-9]+\.", segm):

        if re.match(r"[0-9]+$|[0-9]+\.", segm):
            cnt_digit_segments += 1
            if re.match(r"[0-9]+$", segm):
                pass
            insert_features("segment_[0-9]_" + str(cnt_path_segments) + ":1", features, url_cnt)  # 4b

        if re.match(r"[^\d]+\d+[^\d]+$", segm):
            insert_features("segment_substr[0-9]_" + str(cnt_path_segments) + ":1", features, url_cnt)  # 4c

        if re.match(r"\d+[^\d]", segm):
            insert_features("segment_digits_in_begin_" + str(cnt_path_segments) + ":1", features, url_cnt)

        if re.match(r".*[^\d]\d+$|.*[^\d]\d+\.(\w+)$", segm):
            insert_features("segment_digits_in_end_" + str(cnt_path_segments) + ":1", features, url_cnt)  # 4c

        if re.match(r"[a-zA-Z]+$", segm):
            #insert_features("segment_only_letters_" + str(cnt_path_segments) + ":1", features, url_cnt)
            if (segm != segm.lower()):
                insert_features("segment_not_lower_" + str(cnt_path_segments) + ":1", features, url_cnt)

        match = re.match(r".+\.(.+)", segm)
        if match:
            insert_features("segment_ext_" + str(cnt_path_segments) + ":" + match.group(1).lower(), features,
                            url_cnt)  # 4d
            insert_features("segment_ext_exist:", features, url_cnt)  # 4d
            insert_features("segment_ext:" + match.group(1).lower(), features, url_cnt)  # 4d

        match = re.match(r"[^\d]+\d+[^\d]+\.(\W+)", segm)
        if match:
            insert_features("segment_ext_substr[0-9]_" + str(cnt_path_segments) + ":" + match.group(1).lower(),
                            features, url_cnt)  # 4e

        if re.match(r".*\(.*\).*|.*\(.*\).*\.(\W+)", segm):
            if re.match(r"\(.*\).*|\(.*\).*\.(\W+)", segm):
                insert_features("segment_include_brackets_in_begin" + str(cnt_path_segments) + ":1", features, url_cnt)
            elif re.match(r".*\(.*\)|.*\(.*\)\.(\W+)", segm):
                insert_features("segment_include_brackets_in_end" + str(cnt_path_segments) + ":1", features, url_cnt)
            else:
                insert_features("segment_include_brackets_" + str(cnt_path_segments) + ":1", features, url_cnt)

        match = re.findall(r"[-,;:@_\s]", segm)

        if len(match) > 0:
            if len(match) > 7:
                insert_features("segment_many_special_symbol_" + str(cnt_path_segments) + ":" + str(len(match)),
                                features, url_cnt)
        m = set(match)
        for string in m:
            try:
                string = string.encode("utf-8")
                insert_features("segment_special_symbol_" + str(cnt_path_segments) + ":1", features, url_cnt)  # 4e
                insert_features("segment_special_symbol_" + string + ":1", features, url_cnt)  # 4e
                insert_features("segment_include_cnt_special_symbol_" + string + ":" + str(match.count(string)),
                                features, url_cnt)
            except:
                pass

            if string == ':':
                insert_features("segment_before_:_" + str(cnt_path_segments) + ":" + (re.search(r"(.*):", segm)).group(1),
                                features, url_cnt)

        try:
            segm = segm.decode("utf-8")
            insert_features("segment_len_" + str(cnt_path_segments) + ":" + str(len(segm)), features,
                        url_cnt)  # 4f  !!!!!DELETE
        except:
            pass

        cnt_path_segments += 1

    insert_features("segments:" + str(cnt_path_segments), features, url_cnt)  # 4f
    insert_features("url_cnt_digit_segments" + ":" + str(cnt_digit_segments), features, url_cnt)

    query = urllib.unquote(query.encode("ascii"))
    query_segment = query.split('&')
    # #print query
    # #print path
    if (query == ""):
        insert_features("not_param:", features, url_cnt)
    else:
        insert_features("url_include_param:1", features, url_cnt)  # 4f
        for segm in query_segment:
            if segm == "":
                continue


            match = re.match(r"(.+)=(.+)", segm)
            if match:
                insert_features("param_name:" + match.group(1).lower(), features, url_cnt)
                insert_features("param:" + match.group().lower(), features, url_cnt)
            else:
                insert_features("param_name:" + segm, features, url_cnt)


def extract_features(QLINK_URLS, UNKNOWN_URLS, features):
    url_cnt = 0
    for url in QLINK_URLS:
        parsed = urlparse.urlparse(url)
        add_features(parsed.path, parsed.query, features, url_cnt)
        url_cnt += 1

    for url in UNKNOWN_URLS:
        parsed = urlparse.urlparse(url)
        add_features(parsed.path, parsed.query, features, url_cnt)
        url_cnt += 1


def define_segments(QLINK_URLS, UNKNOWN_URLS, QUOTA):
    global sekitei
    sekitei.quota = QUOTA
    sekitei.cur_quota = 0
    # cnt_clusters = 20
    alpha = 0.07
    border = 0.5 * alpha * len(QLINK_URLS)
    # border = 50
    features = {}
    ALL_URLS = list(QLINK_URLS)
    ALL_URLS.extend(UNKNOWN_URLS)

    extract_features(QLINK_URLS, UNKNOWN_URLS, features)
    # #print
    # for k in features.keys():
    #   #print k + "\t" + str(len(features[k]))

    main_features = ["segments:"]
    for key in features.keys():
        # print key + "\t"+str(len(features[key]))
        not_main = True
        for f in main_features:
            if key.find(f) != -1:
                not_main = False
        if ((len(features[key]) <= border) or (len(features[key]) > (2 * len(QLINK_URLS) - border))) and not_main:
            features.pop(key)

    # #print
    # for k in features.keys():
    #    #print k + str(features[k])

    cnt_features = len(features.keys())

    features_matrix = np.zeros(shape=(len(QLINK_URLS) + len(UNKNOWN_URLS), cnt_features))

    sekitei.features = []
    feat_cnt = 0
    for feat in features.keys():
        sekitei.features.append(feat)
        for url_ind in features[feat]:
            features_matrix[url_ind][feat_cnt] = 1.0
        feat_cnt += 1

    #sekitei.clustering = Birch(threshold=0.5, branching_factor=100, n_clusters=33, compute_labels=True, copy=True)
    sekitei.clustering = KMeans(n_clusters=33)
    res = sekitei.clustering.fit_predict(features_matrix)
    #print sekitei.features

    cnt_clusters = len(np.unique(res))

    res_dict = dict.fromkeys(np.unique(res))

    sekitei.clusters_size = [0 for i in xrange(cnt_clusters)]
    sekitei.qlinks_in_clusters = [0 for i in xrange(cnt_clusters)]

    for i in xrange(len(res)):
        if not (res_dict[res[i]]):
            res_dict[res[i]] = []
        if ALL_URLS[i] in QLINK_URLS:
            sekitei.qlinks_in_clusters[res[i]] += 1
        res_dict[res[i]].append(ALL_URLS[i])
        sekitei.clusters_size[res[i]] += 1

    sekitei.sorted_qlink_and_size_ratio = []
    sekitei.qlink_ratio = []
    sekitei.quota_cluster = []


    for i in xrange(cnt_clusters):
        sekitei.quota_cluster.append(QUOTA * float(sekitei.clusters_size[i]) / (len(QLINK_URLS) + len(UNKNOWN_URLS)))
        if sekitei.clusters_size[i] == 0:
            sekitei.qlink_and_size_ratio.append([0, 0, 0])
            sekitei.qlink_ratio.append(0)
        else:
            sekitei.qlink_ratio.append((float(sekitei.qlinks_in_clusters[i]) / sekitei.clusters_size[i]))
            sekitei.sorted_qlink_and_size_ratio.append([float(sekitei.qlinks_in_clusters[i]) / sekitei.clusters_size[i],
                                                        float(sekitei.clusters_size[i]) / (
                                                        len(QLINK_URLS) + len(UNKNOWN_URLS)),
                                                        float(sekitei.clusters_size[i]) / (
                                                        len(QLINK_URLS) + len(UNKNOWN_URLS))])


    sekitei.sorted_qlink_and_size_ratio.sort(key=operator.itemgetter(0), reverse=True)
    add_ratio = 0;
    for i in xrange(cnt_clusters):
        sekitei.sorted_qlink_and_size_ratio[i][1] += add_ratio
        add_ratio = sekitei.sorted_qlink_and_size_ratio[i][1]

    sekitei.try_fetch = 0
    sekitei.border = 1
    sekitei.cur_cluster = 0
    """
    out = open("./clasters_log", "w")
    for i in res_dict.keys():
        out.write("\n")
        out.write("CLUSTER " + str(i) + "\n")
        for j in res_dict[i]:
            out.write(j + "\n")

    out.close()

    global fetch_log
    fetch_log = open("./fetch_log", "w")


    model = TSNE(n_components=2, random_state=0)
    y = model.fit_transform(features_matrix)
    """
    # #print y
    """
    cm = plt.get_cmap('jet')
    plt.figure(figsize=(15, 15))
    plt.scatter(y[:, 0], y[:, 1], c=map(lambda c: cm(1.0 * c /cnt_clusters), res))
    plt.axis('off')
    plt.show()

    print "____________________________"
    print sekitei.clusters_size
    print sekitei.qlinks_in_clusters
    print sekitei.qlink_ratio

    for i in xrange(len(sekitei.sorted_qlink_and_size_ratio)):
        print sekitei.sorted_qlink_and_size_ratio[i]
    print

    global QLINKS
    QLINKS = []
    with open("./data/urls.lenta.examined") as i_file:
        for line in i_file:
            line = line.strip();
            QLINKS.append(line);
    """

# returns True if need to fetch url
#
def fetch_url(url):
    sekitei.try_fetch += 1
    using_quota = (float(sekitei.cur_quota) / sekitei.quota)
    # ##print using_quota
    for i in xrange(sekitei.cur_cluster, len(sekitei.sorted_qlink_and_size_ratio)):

        if (using_quota <= sekitei.sorted_qlink_and_size_ratio[i][1] + 0.01) \
                and (sekitei.border >= sekitei.sorted_qlink_and_size_ratio[i][0]):
            if (sekitei.try_fetch > 1.8 * sekitei.quota * sekitei.sorted_qlink_and_size_ratio[i][2]) and (
                i < len(sekitei.sorted_qlink_and_size_ratio) - 1):
                sekitei.cur_cluster = i + 1
                border = sekitei.sorted_qlink_and_size_ratio[i + 1][0]
            else:
                sekitei.cur_cluster = i
                border = sekitei.sorted_qlink_and_size_ratio[i][0]
            break

    if sekitei.border != border:
       #print "******************************"
       #print"try_fetch: ", sekitei.try_fetch, "\tusing_quota", using_quota, "\tborder", border, "\tcur_cluster", sekitei.cur_cluster
       sekitei.try_fetch = 0
       sekitei.border = border

    url_features = {}
    l = []
    l.append(url)
    extract_features(l, [], url_features)


    url_x = []

    for feat in sekitei.features:
        if url_features.get(feat):
            url_x.append(1.0)
        else:
            url_x.append(0.0)

    x = np.asarray(url_x).reshape(1, -1)

    label = sekitei.clustering.predict(x)
    """
    fetch_log.write(url + '\n' + "\tCLASTER: " + str(label[0]) + '\n' + "\tCLASTER_RATIO: " + str(
        sekitei.qlink_ratio[label[0]]) + '\n'
                    + "\tIS QLINK: " + str(url in QLINKS) + "\n" + "\tRESPONSE: " + str(
        (sekitei.qlink_ratio[label[0]] >= border)) + '\n')
    if ((url in QLINKS) == False) and (sekitei.qlink_ratio[label[0]] >= border):
        # print "ERROR"
        fetch_log.write("\tERROR! \n")
        fetch_log.write(str(url_features))
        fetch_log.write("\n\t FEATURES_MAIN: \n")
        for feat in sekitei.features:
            if url_features.get(feat):
                fetch_log.write("\t" + feat + "\n")
    """

    if (sekitei.qlink_ratio[label[0]] >= border):
        #sekitei.quota_cluster[label[0]] -= 1
        sekitei.cur_quota += 1
        return True
    # return sekitei.fetch_url(url);
    return False