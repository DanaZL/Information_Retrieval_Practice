
# coding: utf-8
import sys
import os
import re
import random
import time
import sekitei_segments

       
def extract_names(f):
    m = re.search(r'urls\.(\w+)\.(\w+)', f);
    if m is not None:
        return ( m.group(1), m.group(2), f) ;
def read_feas(file_name):
    result = {};
    with open(file_name) as f:
        
        for line in f:
            line = line.strip();
            p = line.split('\t');
            if len(p) >1: result[p[0]] = p[1];
    return result            
                
def compare_results(result, pattern):
    equals =0.
    for p in pattern.keys():
         if p in result:
            equals += 1
    return equals/ len(pattern);

def shufle_urls_from_file(FILE_NAME):
    urls = [];
    with open(FILE_NAME ) as i_file:
         for line in i_file:
            line = line.strip();
            urls.append(line);
    random.shuffle(urls)
    return urls;

#
# paths
#
INPUT_PATH='./data/'
CHECK_PATH='./check/'

if os.path.exists(INPUT_PATH) != True:
    print >> sys.stderr, "Missing input path " + INPUT_PATHT;
    sys.exit(1);
    
if os.path.exists(CHECK_PATH) != True:
    print >> sys.stderr, "Missing check path " + CHECK_PATH;
    sys.exit(1);
    
files = os.listdir(INPUT_PATH);
files = sorted(files)
names = map(extract_names, files)

#
# defines 
#
count = len(names) / 2
curl_for_segments = 500;
MAX_QUOTA = 10000
result = [];
MAX_TIME = 15.;
MIN_RATE = 0.7

#
# check loop
#
for i in range(0, count):
#for i in range(0,  1):
    name = names[i*2][0]
    print "Run test: " + names[i*2][0]
    
    
    f1 = INPUT_PATH + names[i*2][2]
    qlink_urls = shufle_urls_from_file(f1);    
    f2 = INPUT_PATH + names[i*2+1][2]
    unk_urls = shufle_urls_from_file(f2);
            
    t1 = time.time()

    # define segments here
    sekitei_segments.define_segments(qlink_urls[:curl_for_segments],
                                     unk_urls[:curl_for_segments],
                                     MAX_QUOTA);
    qlink_urls = [(1, url) for url in qlink_urls[curl_for_segments:]];
    unk_urls = [(0, url) for url in unk_urls[curl_for_segments:] ];
    
    urls_mix = []
    urls_mix.extend(qlink_urls);
    urls_mix.extend(unk_urls);
    random.shuffle(urls_mix)
    
    urls_fetched = 0;
    qlinks_fetched = 0    
    qlinks_count = len(qlink_urls);


    cnt_fetching = 0
    print "ALL URL FOR FETCHING: " + str(len(urls_mix))
    for url in urls_mix:
        cnt_fetching += 1
        if sekitei_segments.fetch_url(url[1]) :
            urls_fetched +=1;
            qlinks_fetched += url[0];
        if urls_fetched >= MAX_QUOTA:
            break;
    t2 = time.time()
    print "try fetched URL: " + str(cnt_fetching)
    result.append((name, (t2-t1), float(urls_fetched), float(qlinks_fetched), float(qlinks_count) ) );

print "=== Test summary ===";
out= ''
avg_qlink_rate = 0
avg_fetched = 0;
for r in result:    
    out = r[0] + " t: " + str(r[1]) + "  fetched rate: " + str(r[2]/MAX_QUOTA)                 + " qlink rate:" + str(r[3] / r[4] )
        
    if r[1] > MAX_TIME:
        out += " : fail due execution time" 
    else :
        avg_qlink_rate += r[3] / r[4] 
        avg_fetched += r[2]/MAX_QUOTA;
    
    print out;
    
avg_qlink_rate /= len(result);
avg_fetched /= len(result);
print "\naverage fetched rate: " + str(avg_fetched) + " qlink rate: " + str(avg_qlink_rate);
f1 =  (2 * avg_qlink_rate * avg_fetched) / (avg_qlink_rate + avg_fetched )
print "\nF1 - score: " + str(f1);
print ""

if f1 < MIN_RATE:
    print "YOUR RESULT SCORE: 0";
else:    
    if f1 > 0.9:
        print "YOUR RESULT SCORE: 5";
    else:
        if f1 > 0.8:
            print "YOUR RESULT SCORE: 4";
        else:
            if f1 > 0.7:
                print "YOUR RESULT SCORE: 3";





