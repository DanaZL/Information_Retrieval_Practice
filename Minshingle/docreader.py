#!/usr/bin/env python

import document_pb2
import struct
import gzip
import sys


class DocumentStreamReader:
    def __init__(self, path):
        if path.endswith('.gz'):
            self.stream = gzip.open(path, 'rb')
        else:
            self.stream = open(path, 'rb')

    def __iter__(self):
        while True:
            sb = self.stream.read(4)
            if sb == '':
                return

            size = struct.unpack('i', sb)[0]
            msg = self.stream.read(size)
            doc = document_pb2.document()
            doc.ParseFromString(msg)
            yield doc


def main():
    """
    Every 'document' is a protobuf with 3 fields: url, raw text and already extracted _clean_ text
    You should use clean text for shingling
    """
    reader = DocumentStreamReader(sys.argv[1:])
    for doc in reader:
        print "%s\tbody: %d, text: %d" % (
            doc.url,
            len(doc.body) if doc.HasField('body') else 0,
            len(doc.text) if doc.HasField('text') else 0
        )


if __name__ == '__main__':
    main()
