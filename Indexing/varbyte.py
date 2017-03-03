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


def compress(dl):
    bs = BitstreamWriter()

    for idx in xrange(0, len(dl)):
        if (idx == 0):
            dif = dl[idx]
        else:
            dif = dl[idx] - dl[idx - 1]
        cnt_bit = 0

        while dif != 0:
            bit = dif % 2
            dif = dif / 2
            bs.add(bit)
            cnt_bit += 1

            if cnt_bit == 7:
                cnt_bit = 0
                if (dif):
                    bs.add(0)
                else:
                    bs.add(1)

        if cnt_bit > 0:
            while cnt_bit < 7:
                bs.add(0)
                cnt_bit += 1
            bs.add(1)

    return bs.getbytes()


def decompress(s):
    bs = BitstreamReader(s)
    dl = []
    cnt = 0
    while not bs.finished():
        res = 0
        degree = 1
        while True:
            for i in range(1, 8):
                res += degree * bs.get()
                degree *= 2

            if bs.get():
                break

        if cnt == 0:
            dl.append(res)
        else:
            dl.append(res + dl[cnt - 1])
        cnt += 1

    return dl

