"""Convert an immutable gzip pack to gzip NDJSON with bounded entry memory."""
import gzip,json,zlib

class Reader:
    def __init__(self,f):self.f=f;self.buffer='';self.decoder=json.JSONDecoder();self.eof=False
    def more(self):
        # Retry JSON decoding per MiB rather than per64KiB: large immutable
        # manifest objects otherwise cause repeated scans of their entire prefix.
        chunk=self.f.read(1024*1024)
        if not chunk:self.eof=True
        self.buffer+=chunk
    def space(self):
        while True:
            self.buffer=self.buffer.lstrip()
            if self.buffer or self.eof:return
            self.more()
    def expect(self,c):
        self.space()
        if not self.buffer.startswith(c):raise ValueError('Malformed research pack delimiter')
        self.buffer=self.buffer[len(c):]
    def peek(self):self.space();return self.buffer[:1]
    def value(self):
        self.space()
        while True:
            try:v,end=self.decoder.raw_decode(self.buffer);self.buffer=self.buffer[end:];return v
            except json.JSONDecodeError:
                if self.eof:raise ValueError('Truncated research pack')
                if len(self.buffer)>32*1024*1024:raise ValueError('Research pack entry exceeds bounded parser limit')
                self.more()

def iter_records(path):
    """The producer writes header fields then entries last. No full-pack string."""
    with gzip.open(path,'rt',encoding='utf-8') as f:
        r=Reader(f);r.expect('{');head={}
        while True:
            key=r.value();r.expect(':')
            if key=='entries':break
            head[key]=r.value();r.expect(',')
        if head.get('schema')!='datapack_extra.v1':raise ValueError('Not an extra pack')
        yield {**head,'schema':'datapack_extra_stream.v1'}
        r.expect('{');n=0
        while r.peek()!='}':
            key=r.value();r.expect(':');value=r.value()
            if not isinstance(key,str) or not key.startswith('/api/'):raise ValueError('Invalid entry path')
            yield {'path':key,'data':value};n+=1
            if r.peek()=='}':break
            r.expect(',')
        r.expect('}');r.expect('}')
        if r.peek():raise ValueError('Unexpected trailing research pack content')
        yield {'complete':True,'n_entries':n,'version':head['version']}

def gzip_records(path):
    compressor=zlib.compressobj(level=1,wbits=31)
    for record in iter_records(path):
        raw=(json.dumps(record,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')
        for i in range(0,len(raw),65536):
            out=compressor.compress(raw[i:i+65536])
            if out:yield out
    final=compressor.flush()
    if final:yield final
