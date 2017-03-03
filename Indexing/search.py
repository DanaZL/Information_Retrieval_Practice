# -*- coding: utf-8 -*-

import pickle
import sys
import re
import varbyte
import mmh3
import simple9

SPLIT_RGX = re.compile(r'\w+|[\(\)&\|!]', re.U)

class QtreeTypeInfo:
    def __init__(self, value, op=False, bracket=False, term=False):
        self.value = value
        self.is_operator = op
        self.is_bracket = bracket
        self.is_term = term

    def __repr__(self):
        return repr(self.value)

    def __eq__(self, other):
        if isinstance(other, QtreeTypeInfo):
            return self.value == other.value
        return self.value == other


class QTreeTerm(QtreeTypeInfo):
    def __init__(self, term):
        QtreeTypeInfo.__init__(self, term, term=True)


class QTreeOperator(QtreeTypeInfo):
    def __init__(self, op):
        QtreeTypeInfo.__init__(self, op, op=True)
        self.priority = get_operator_prio(op)
        self.left = None
        self.right = None


class QTreeBracket(QtreeTypeInfo):
    def __init__(self, bracket):
        QtreeTypeInfo.__init__(self, bracket, bracket=True)


def get_operator_prio(s):
    if s == '|':
        return 0
    if s == '&':
        return 1
    if s == '!':
        return 2

    return None


def is_operator(s):
    return get_operator_prio(s) is not None


def tokenize_query(q):
    # print "tokenize", q
    tokens = []
    for t in map(lambda w: w.encode('utf-8'), re.findall(SPLIT_RGX, q)):
        if t == '(' or t == ')':
            tokens.append(QTreeBracket(t))
        elif is_operator(t):
            tokens.append(QTreeOperator(t))
        else:
            tokens.append(QTreeTerm(t))

    return tokens


def build_query_tree(tokens):
    if len(tokens) == 1:
        return tokens[0]
    else:
        tokens[1].left = tokens[0]
        tokens[1].right = build_query_tree(tokens[2:])
        return tokens[1]


def parse_query(q):
    tokens = tokenize_query(q)
    #print tokens
    return build_query_tree(tokens)

def intersection(term1, term2):
    res = []
    idx1, idx2 = 0, 0

    try:
        doc_list1 = encoder.decompress(index[abs(mmh3.hash(term1))])
    except:
        doc_list1 = term1
    try:
        doc_list2 = encoder.decompress(index[abs(mmh3.hash(term2))])
    except:
        doc_list2 = term2

    return sorted(list(set(doc_list1) & set(doc_list2)))


def search(root):
    if root.is_term:
        return encoder.decompress(index[abs(mmh3.hash(root.value))])
    return intersection(search(root.left), search(root.right))


file1 = open("./index", "r")
encoder_str = file1.readline()

if encoder_str[:-1] == 'varbyte':
    encoder = varbyte
elif encoder_str[:-1] == 'simple9':
    encoder = simple9

index = pickle.load(file1)
url_list = pickle.load(file1)
file1.close()

# print index

while True:
    try:
        line = raw_input()
        tree = parse_query(line.decode('utf-8').lower())
        res = search(tree)
        print line
        print len(res)
        for id in res:
            print url_list[id - 1]
    except:
        break




