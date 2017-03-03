# -*- coding: utf-8 -*-

class BitstreamWriter:
    def __init__(self):
        self.nbits  = 0
        self.curbyte = 0
        self.vbytes = []

    """ add single bit """
    def add(self, x):
        self.curbyte |= x << (8-1 - (self.nbits % 8))
        self.nbits += 1

        if self.nbits % 8 == 0:
#             print self.curbyte
            self.vbytes.append(chr(self.curbyte))
            self.curbyte = 0

    """ get byte-aligned bits """
    def getbytes(self):
        if self.nbits & 7 == 0:
            return "".join(self.vbytes)

        return "".join(self.vbytes) + chr(self.curbyte)


class BitstreamReader:
    def __init__(self, blob):
        self.blob = blob
        self.pos  = 0

    """ extract next bit """
    def get(self):
        ibyte = self.pos / 8
        ibit  = self.pos & 7

        self.pos += 1
        return (ord(self.blob[ibyte]) & (1 << (7 - ibit))) >> (7 - ibit)

    def finished(self):
        return self.pos >= len(self.blob) * 8


variants = {1:28, 2:14, 3:9, 4:7, 5:5, 6:4, 7:3, 8:2, 9:1}
code_var = {28:1, 14:2, 9:3, 7:4, 5:5, 4:6, 3:7, 2:8, 1:9}

payload = 28
code = 4

def cnt_bit(x):
    if x < 2**1:
        return 1
    
    if x < 2**2:
        return 2
    
    if x < 2**3:
        return 3
    
    if x < 2**4:
        return 4
    
    if x < 2**5:
        return 5
    
    if x < 2**7:
        return 7
    
    if x < 2**9:
        return 9
    
    if x < 2**14:
        return 14
    
    if x < 2**28:
        return 28
    
def bs_push(bs, x, cnt_bit):
    while x != 0:
        bit = x % 2
        x = x / 2
        bs.add(bit)
        cnt_bit -= 1
                
    if cnt_bit > 0:
        while cnt_bit != 0:
            bs.add(0)
            cnt_bit -= 1


def compress(dl):
    bs = BitstreamWriter()
    curr_payload = 0
    curr_cnt = 0
    for idx in xrange(0, len(dl)):
        
        if (idx == 0):
            dif = dl[idx]
        else:
            dif = dl[idx] - dl[idx - 1]
        
        need_bit = cnt_bit(dif)
        
        if need_bit == curr_payload:
            bs_push(bs, dif, need_bit)
            curr_cnt += 1
            if curr_cnt == (payload / need_bit):
                bs_push(bs, 0, 28 - curr_cnt * curr_payload)
                curr_payload = 0
                curr_cnt = 0
                
        elif curr_cnt == 0:
            curr_payload = need_bit
            bs_push(bs, code_var[curr_payload], code)
            bs_push(bs, dif, need_bit)
            curr_cnt += 1
#             print curr_cnt, curr_payload
            
            if curr_cnt == (payload / need_bit):
                bs_push(bs, 0, 28 - curr_cnt * curr_payload)
                curr_payload = 0
                curr_cnt = 0
                
        elif curr_cnt != 0:
            bs_push(bs, 0, 28 - curr_cnt * curr_payload)
            
            curr_payload = need_bit
            bs_push(bs, code_var[curr_payload], code)
            bs_push(bs, dif, need_bit)
            curr_cnt = 1
            
            if curr_cnt == (payload / need_bit):
                bs_push(bs, 0, 28 - curr_cnt * curr_payload)
                curr_payload = 0
                curr_cnt = 0
                
        if idx == len(dl) - 1 and curr_cnt != 0:
            bs_push(bs, 0, 28 - curr_cnt * curr_payload)
            
                
    return bs.getbytes()

def bs_pop(bs, cnt_bit):
    res = 0
    degree = 1
    for i in range(0, cnt_bit):
#         print "i", i
        res += degree * bs.get()
        degree *= 2
    
    return res
    

def decompress(s):
    bs = BitstreamReader(s)
    dl = []
    cnt = 0
    while not bs.finished():
        curr_cnt = 0
        curr_code = bs_pop(bs, 4)
        curr_payload = variants[curr_code]
#         print "cp", curr_payload
        for i in range(0, payload / curr_payload):
#             print "n", i
            res = bs_pop(bs, curr_payload)
            curr_cnt += 1
            if res != 0:
                if cnt == 0:
                    dl.append(res)
#                     print "RES", res
                    cnt += 1
                else:
#                     print "RES", res
                    dl.append(res + dl[cnt-1])
                    cnt += 1
        res = bs_pop(bs, 28 - curr_payload * curr_cnt)
                
        
    return dl
