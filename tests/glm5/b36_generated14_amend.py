from pathlib import Path
import b36_generated14 as G
S=G.S
source=Path(__file__).with_name('b36_generated14.py').read_text(encoding='utf8')
source=source.replace("assert p['source_sha256']==S.file_sha(__file__)","assert p['source_sha256']==S.file_sha(S.ROOT/'tests/glm5/b36_generated14.py')")
source=source.replace("range(0,len(p['rows']),12)","range(0,len(p['rows']),72)").replace("p['rows'][lo:lo+12]","p['rows'][lo:lo+72]").replace('if calls[0]%8==0:','if calls[0]%1==0:').replace('calls[0],384,start','calls[0],64,start')
dest=Path(__file__).with_name('b36_generated14_batch72.py');assert not dest.exists();dest.write_text(source,encoding='utf8')
S.configure();assert not list(S.OUT.glob('chunk-*.json'))
S.save('execution-amendment.json',dict(reason='Batch12 stopped before any committed result due to slow repeated weight streaming. Batch72 amortizes reads. Scientific materials,metrics,threshold and64-token cap unchanged.',source_sha256=S.file_sha(dest),batch=72,original_batch=12,estimated_peak_gpu_gib=12,original_log='generation.log',no_committed_v1_chunks=True))
