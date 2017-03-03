
# coding: utf-8
#
#
# Test script to check quality of features extraction
# Directory structure:
# ./data/ <- folder with urls data files
# ./check/ <- folder with files included correctly extracted features
# This script will scan ./data/ folder and fetch urls from files
# each site was placed to pair of files general and examined.
# Each pair of files for site will be passed as parameter of 
# the extract_features module. 
###########
# Rules:
# 1. Execution of single feature extraction from coupe of site files limits by 1 sec 
# 1. Features diff no more then: 0.05
# 1, Features diff by count: no more then 0.1
#######
#

import sys
import os
import re
import time
import extract_features


def extract_names(f):
    m = re.search(r'urls\.(\w+)\.(\w+)', f);
    if m is not None:
        return ( m.group(1), m.group(2), f);
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
            equals += 1.
    return equals/ len(pattern);
            
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
#print names 

count = len(names) / 2
boot_strap_cnt = 100;
test_result = {};

for bs in range(0, boot_strap_cnt):
    #print "Bootstrap step: " + str(bs); 
    for i in range(0, count):
        #print "\tProcessing: " + names[i*2][0];
        f1 = INPUT_PATH + names[i*2][2]
        f2 = INPUT_PATH + names[i*2+1][2]
        name = names[i*2][0];
        out = CHECK_PATH + name + '.fea.res';
        check_f = CHECK_PATH + name + '.fea';
        if name not in test_result:
            test_result[name] = (0,0,0, False);
        if os.path.exists(check_f) != True: 
            print >> sys.stderr, "Output test file was not created";
            continue;
        if os.path.exists(out):
            os.remove(out);

        t1 = time.time() 
        extract_features.extract_features(f1, f2, out);
        t2 = time.time()
        if os.path.exists(out)!= True: 
            print >> sys.stderr, "Output test file was not created";
            continue;
        result = read_feas(out)
        pattern = read_feas(check_f)
        t = test_result[name][0] +(t2-t1);
        w = test_result[name][1] + compare_results(result, pattern);
        c = test_result[name][2] + float(len(result)) / float(len(pattern));
    

        test_result[name] = (t, w, c , True );
        
count_fails=0;    
for name in test_result.keys():
    out = name + " test: ( err: " + str(1-(test_result[name][1]/boot_strap_cnt)) + ") ";
    if test_result[name][0]/boot_strap_cnt > 1:
        out += " fail due execution time. Average execution time: " + str(test_result[name][0]/boot_strap_cnt)
        count_fails+=1;
    else:
        if test_result[name][1]/boot_strap_cnt < 0.95:
            out += " fail due features names. Err is more then 0.05"
            count_fails+=1;
        else :
            if test_result[name][2]/ boot_strap_cnt > 1.1:
                out += " fail due count of features: " + str(test_result[name][2]/ boot_strap_cnt);
                count_fails += 1;
            else :
                if test_result[name][3] == False:
                    out += " fail due system errors";
                    count_fails +=1;
                else :
                    out += 'passed';
    print out; 

if count_fails > 0: 
    print 'Total fails: ' + str(count_fails) + " NOT PASSED";
else : 
    print 'PASSED';



