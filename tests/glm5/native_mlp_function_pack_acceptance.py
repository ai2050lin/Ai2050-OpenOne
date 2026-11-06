"""Read the entire real offline stream with bounded memory and verify every CAS blob."""
import sys,json,zlib,codecs,time,hashlib
from pathlib import Path
import requests
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
from deploy.research_library import canonical

def main():
    reg=P.read(P.OUT/'delivery-registration.json');expected={reg[k] for k in ['report_sha256','graph_sha256','case_index_sha256','acceptance_sha256','scientific_memo_sha256']};found=set();count=0;cas=0;maxline=0;header=None;footer=None;buffer='';total=0;started=time.time();decomp=zlib.decompressobj(31);decoder=codecs.getincrementaldecoder('utf-8')();space=False
    def line(text):
        nonlocal count,cas,maxline,header,footer,space
        if not text:return
        maxline=max(maxline,len(text));d=json.loads(text)
        if header is None:assert d['schema']=='datapack_extra_stream.v1' and d['id']=='research-library';header=d;return
        if d.get('complete'):assert d['n_entries']==count and d['version']==header['version'];footer=d;return
        assert footer is None;count+=1;path=d['path'];v=d['data']
        if path.startswith('/api/research-library/artifacts/'):
            sha=path.rsplit('/',1)[-1];assert hashlib.sha256(canonical(v)).hexdigest()==sha;cas+=1
            if sha in expected:found.add(sha)
        if path=='/api/traces/research-space':space=True;assert any(s['run_id']==P.RUN and s['report_sha256']==reg['report_sha256'] for s in v['studies'])
    url='http://127.0.0.1:5001/api/platform/datapack/extra/research-library?format=ndjson'
    with requests.get(url,stream=True,timeout=(30,180)) as response:
        response.raise_for_status();assert response.headers['Content-Type'].startswith('application/x-ndjson+gzip')
        for chunk in response.iter_content(131072):
            total+=len(chunk);buffer+=decoder.decode(decomp.decompress(chunk))
            while '\n' in buffer:
                x,buffer=buffer.split('\n',1);line(x)
        buffer+=decoder.decode(decomp.flush(),final=True)
        if buffer.strip():line(buffer)
    assert decomp.eof and footer and expected==found and space
    result={'kind':'real','run_id':P.RUN,'version':header['version'],'entries':count,'CAS_blobs_hash_verified':cas,'all_current_report_graph_case_acceptance_memo_present':True,'research_space_present':space,'compressed_bytes_streamed':total,'max_one_entry_characters':maxline,'whole_pack_string_materialized':False,'seconds':time.time()-started}
    P.save(P.OUT/'offline-stream-acceptance.json',result);print(json.dumps(result))
if __name__=='__main__':main()
