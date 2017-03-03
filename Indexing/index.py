# -*- coding: utf-8 -*-

import docreader
import doc2words
import varbyte
import simple9
import pickle
import mmh3
import json

#urls = docreader.DocumentStreamReader(["../dataset/lenta.ru_4deb864d-3c46-45e6-85f4-a7ff7544a3fb_01.gz"])
arg = docreader.parse_command_line().files
reader = docreader.DocumentStreamReader(arg[1:])
encoder_str = arg[0]

if encoder_str == 'varbyte':
    encoder = varbyte
elif encoder_str == 'simple9':
    encoder = simple9

#for i in urls:
#    print i.text.encode("utf-8")
#    break

term_dictionary = {}
url_list = []

doc_id = 0
for url in reader:
    doc_id += 1
    url_list.append(url.url)
    words = doc2words.extract_words(url.text)
    uniq_words = list(set(words))

    for word in uniq_words:
        #print mmh3.hash()
        hash = abs(mmh3.hash(word.encode("utf-8")))
        if (term_dictionary.get(hash)):
            term_dictionary[hash].append(doc_id)
        else:
            term_dictionary[hash] = []
            term_dictionary[hash].append(doc_id)


print term_dictionary[abs(mmh3.hash("энергоносители"))]
for key in term_dictionary:
    term_dictionary[key] = encoder.compress(term_dictionary[key])
file = open("./index", "w")
file.write(encoder_str + '\n')
pickle.dump(term_dictionary, file)
pickle.dump(url_list, file)
print len(url_list)
file.close()