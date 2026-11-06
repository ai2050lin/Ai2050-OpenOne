from pathlib import Path
p=Path(__file__).resolve().parents[2]/'frontend/src/components/v8/MlpFidelityReport.jsx';s=p.read_text(encoding='utf-8')
changes={
 '此数量不是破解进度。</p><table':'此数量不是破解进度。</p><details><summary>展开全部36层的保真度、预算与失败</summary><table',
 '</tbody></table></>}<details><summary>开发比较':'</tbody></table></details></>}<details><summary>开发比较',
 "const [tab,setTab]=useState('result');":"const [tab,setTab]=useState('result'),[tailOpen,setTailOpen]=useState(false);",
 '<details><summary>真实原场、每题结果与反例</summary><TailExplorer parts={tails}/></details>':'<details onToggle={e=>setTailOpen(e.currentTarget.open)}><summary>真实原场、每题结果与反例</summary>{tailOpen&&<TailExplorer parts={tails}/>}</details>'}
for a,b in changes.items():assert a in s,a;s=s.replace(a,b)
p.write_text(s,encoding='utf-8')
