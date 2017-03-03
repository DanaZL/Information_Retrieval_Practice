#!/usr/bin/env python

"""
This just a draft for homework 'near-duplicates'
Use MinshinglesCounter to make result closer to checker
"""

import sys
import re
import mmh3
import numpy
from docreader import DocumentStreamReader

shingles_dict = {}
url_eq = {}
urls = {}
# urls_intersect

def broder_merge():
    for key, value in shingles_dict.iteritems():
        value.sort()
        cnt = 0
        for url1 in value:
            cnt += 1
            for url2 in value[cnt:]:
                urls_intersect[urls[url1], urls[url2]] += 1


def result_definition(minshingle_dim):
    #cnt_res = 0
    cnt = 0
    urls_str = sorted(urls.keys())
    for url1 in urls_str:
        cnt += 1
        for url2 in urls_str[cnt:]:
            value = urls_intersect[urls[url1], urls[url2]]
            JC = (1.0 * value) / (value + (minshingle_dim - value) * 2)
            if JC >= 0.8:
                #cnt_res +=1
                print url1 + " " + url2 + " " + str(JC)

    #print cnt_res


class MinshinglesCounter:
    SPLIT_RGX = re.compile(r'\w+', re.U)

    def __init__(self, window=5, n=20):
        self.window = window
        self.n = n

    def count(self, text, url):
        words = MinshinglesCounter._extract_words(text)
        shs = self._count_shingles(words)
        mshs = self._select_minshingles(shs)

        res = None
        if len(mshs) == self.n:
            res = mshs
            for sh in res:
                shingles_dict.setdefault(sh, []).append(url)
            return res

        shs = list(set(shs))
        if len(shs) >= self.n:
            res = sorted(shs)[0:self.n]
            for sh in res:
                shingles_dict.setdefault(sh, []).append(url)
            return res

        return res

    def _select_minshingles(self, shs):
        buckets = [None]*self.n
        for x in shs:
            bkt = x % self.n
            buckets[bkt] = x if buckets[bkt] is None else min(buckets[bkt], x)

        return filter(lambda a: a is not None, buckets)

    def _count_shingles(self, words):
        shingles = []
        for i in xrange(len(words) - self.window):
            h = mmh3.hash(' '.join(words[i:i+self.window]).encode('utf-8'))
            shingles.append(h)
        return sorted(shingles)

    @staticmethod
    def _extract_words(text):
        words = re.findall(MinshinglesCounter.SPLIT_RGX, text)
        return words


def main():
    minshingle_dim = 20
    mhc = MinshinglesCounter()
    ind_url = 0
    for path in sys.argv[1:]:
        for doc in DocumentStreamReader(path):
            if doc.url not in urls:
                urls[doc.url] = ind_url
                ind_url += 1;
                mhc.count(doc.text, doc.url)
                #print "%s (text length: %d, minhashes: %s)" % (doc.url, len(doc.text), mhc.count(doc.text, doc.url))


    global urls_intersect
    urls_intersect = numpy.zeros((len(urls), len(urls)))
    broder_merge()
    result_definition(minshingle_dim)


    """
    You may examine content of given files this way (as example):

    for path in sys.argv[1:]:
        for doc in DocumentStreamReader(path):
            print "%s (text length: %d, minhashes: %s)" % (doc.url, len(doc.text), mhc.count(doc.text))
    """

    """
    Write your actual code here.
    Good luck!
    """


if __name__ == '__main__':
    main()
