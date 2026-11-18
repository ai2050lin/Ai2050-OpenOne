"""Storage-safe BF16 reference backend for v2 after Windows auto-dispatch exited."""
import sys
import time
from pathlib import Path

import condition_chain_chat as C
import condition_chain_run as R
import condition_chain_materials as M


class StreamEngine(R.Engine):
    def __init__(self,key):
        import torch
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        started=time.monotonic()
        self.key=key
        folder=R.ROOT/"models/hf"/M.MODELS[key]
        config=AutoConfig.from_pretrained(folder,local_files_only=True)
        with torch.device("meta"):
            self.model=AutoModelForCausalLM.from_config(config,dtype=torch.bfloat16,attn_implementation="sdpa").eval()
        self.tok=AutoTokenizer.from_pretrained(folder,local_files_only=True)
        self.tok.padding_side="left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id=config.pad_token_id
        self.letters=[self.tok.encode(" "+letter,add_special_tokens=False) for letter in "ABCD"]
        assert all(len(tokens)==1 for tokens in self.letters)
        self.letters=[tokens[0] for tokens in self.letters]
        self.dimension=config.hidden_size
        self.layers=self.model.model.layers
        self.handles=[]
        weight_map=R.load(folder/"model.safetensors.index.json")["weight_map"]
        counts = {n:p.numel()*2 for n,p in self.model.named_parameters()}
        budget=8*(2**30)
        static_names=["model.embed_tokens.weight","model.norm.weight","lm_head.weight"]
        used=sum(counts[n] for n in static_names)
        self.placement={}

        def read(keys):
            sources={}
            for n in keys:
                sources.setdefault(weight_map[n],[]).append(n)
            state={}
            for filename,names in sources.items():
                with safe_open(folder/filename,framework="pt",device="cpu") as tensors:
                    for name in names:
                        x=tensors.get_tensor(name)
                        assert x.dtype==torch.bfloat16 and x.numel()*2==counts[name]
                        state[name]=x.to("cuda")
            return state

        static=read(static_names)
        for name,value in static.items():
            self.model.get_submodule(name.rsplit(".",1)[0]).weight=torch.nn.Parameter(value,requires_grad=False)
            self.placement[name]="cuda:0"
        del static
        rotary_type=type(self.model.model.rotary_emb)
        self.model.model.rotary_emb=rotary_type(config,device="cpu").cuda()
        for i,layer in enumerate(self.layers):
            prefix=f"model.layers.{i}."
            names=[f"{prefix}{suffix}" for suffix in layer.state_dict()]
            cost=sum(counts[n] for n in names)

            def load_layer(module,args,names=names,prefix=prefix):
                module.load_state_dict({n.removeprefix(prefix):value for n,value in read(names).items()},assign=True,strict=True)

            def free_layer(module,args,result):
                module.to_empty(device="meta")

            if used+cost<=budget:
                load_layer(layer,None)
                self.placement[f"block{i}"]="cuda:0"
                used+=cost
            else:
                self.handles.extend([layer.register_forward_pre_hook(load_layer),layer.register_forward_hook(free_layer)])
                self.placement[f"block{i}"]="safetensors → cuda:0 → meta"
            if i%4==3 or i+1==len(self.layers):
                R.stage_log(key,"stream-bootstrap",i+1,len(self.layers),started,resident_weight_bytes=used)

    def close(self):
        for handle in self.handles:
            handle.remove()
        super().close()


if __name__=="__main__":
    key=sys.argv[1]
    assert key in ("q14","glm")
    R.setup()
    p=R.protocol()
    R.write(f"{key}-stream-adapter.json",dict(plan_sha=R.file_sha(R.OUT/"plan.json"),
                                            adapter_sha=R.file_sha(Path(__file__)),reason="auto-dispatch process exited before model ready",
                                            altered="weight placement only",frozen_resolution="same material, BF16, SDPA, readouts, scoring and batch size"))
    R.Engine=StreamEngine
    C.run(key)
