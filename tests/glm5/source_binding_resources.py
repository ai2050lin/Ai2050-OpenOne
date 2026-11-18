"""Read-only Windows process GPU residency and system commit telemetry.

DXGI ABI is from Microsoft's dxgi1_4.h. NON_LOCAL usage is reported as
process shared GPU memory, separately from torch allocator figures.
"""
import ctypes as c
from ctypes import wintypes as w
import os,uuid

class GUID(c.Structure):
    _fields_=[('a',w.DWORD),('b',w.WORD),('d',w.WORD),('e',c.c_ubyte*8)]
    @classmethod
    def of(cls,s):return cls.from_buffer_copy(uuid.UUID(s).bytes_le)

class DESC(c.Structure):
    _fields_=[('Description',w.WCHAR*128),('VendorId',w.UINT),('DeviceId',w.UINT),('SubSysId',w.UINT),('Revision',w.UINT),
              ('DedicatedVideoMemory',c.c_size_t),('DedicatedSystemMemory',c.c_size_t),('SharedSystemMemory',c.c_size_t),
              ('LuidLow',w.DWORD),('LuidHigh',w.LONG),('Flags',w.UINT)]

class MEM(c.Structure):
    _fields_=[(s,c.c_uint64) for s in ['Budget','CurrentUsage','AvailableForReservation','CurrentReservation']]

class PERF(c.Structure):
    _fields_=[('cb',w.DWORD)]+[(x,c.c_size_t) for x in ['CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize']]+[(x,w.DWORD) for x in ['HandleCount','ProcessCount','ThreadCount']]

def call(ptr,index,restype,types,*args):
    address=c.cast(ptr,c.POINTER(c.POINTER(c.c_void_p))).contents[index]
    return c.WINFUNCTYPE(restype,c.c_void_p,*types)(address)(ptr,*args)

class GPUMemory:
    def __init__(self):
        self.ptr=None;self.name=None;self.failure=None
        try:
            fn=c.windll.dxgi.CreateDXGIFactory1;fn.argtypes=[c.POINTER(GUID),c.POINTER(c.c_void_p)];fn.restype=w.LONG
            factory=c.c_void_p();iid=GUID.of('770aae78-f26f-4dba-a829-253c83d1b387')
            assert fn(c.byref(iid),c.byref(factory))==0
            try:
                for i in range(16):
                    adapter=c.c_void_p()
                    hr=call(factory,12,w.LONG,[w.UINT,c.POINTER(c.c_void_p)],i,c.byref(adapter))
                    if hr!=0:break
                    try:
                        desc=DESC();assert call(adapter,10,w.LONG,[c.POINTER(DESC)],c.byref(desc))==0
                        if desc.VendorId!=0x10de:continue
                        out=c.c_void_p();iid=GUID.of('645967a4-1392-4310-a798-8053ce3e93fd')
                        assert call(adapter,0,w.LONG,[c.POINTER(GUID),c.POINTER(c.c_void_p)],c.byref(iid),c.byref(out))==0
                        self.ptr=out;self.name=desc.Description;break
                    finally:call(adapter,2,w.ULONG,[])
            finally:call(factory,2,w.ULONG,[])
            assert self.ptr,'NVIDIA DXGI adapter not found'
        except Exception as e:self.failure=repr(e)

    def read(self):
        if self.ptr is None:return {'gpu_telemetry_error':self.failure,'shared_gpu_bytes':None}
        values={}
        for i,name in [(0,'dedicated'),(1,'shared')]:
            m=MEM();hr=call(self.ptr,14,w.LONG,[w.UINT,w.UINT,c.POINTER(MEM)],0,i,c.byref(m))
            if hr!=0:return {'gpu_telemetry_error':hex(hr&0xffffffff),'shared_gpu_bytes':None}
            values[name+'_gpu_bytes']=m.CurrentUsage;values[name+'_gpu_budget_bytes']=m.Budget
        return {'gpu_adapter':self.name,'gpu_telemetry':'DXGI IDXGIAdapter3 process residency',**values}

    def __del__(self):
        if self.ptr:
            try:call(self.ptr,2,w.ULONG,[])
            except Exception:pass

_gpu=None
def resources():
    import psutil
    global _gpu
    p=PERF();p.cb=c.sizeof(p);assert c.windll.psapi.GetPerformanceInfo(c.byref(p),p.cb)
    if _gpu is None:_gpu=GPUMemory()
    return {'pid':os.getpid(),'ram_available_bytes':psutil.virtual_memory().available,
            'commit_available_bytes':(p.CommitLimit-p.CommitTotal)*p.PageSize,
            'commit_used_bytes':p.CommitTotal*p.PageSize,**_gpu.read()}

if __name__=='__main__':
    import json,torch
    print(json.dumps({'before_cuda':resources()}),flush=True)
    x=torch.ones(256*1024*1024,dtype=torch.float16,device='cuda');torch.cuda.synchronize()
    print(json.dumps({'with_512MiB_tensor':resources(),'torch_allocated':torch.cuda.memory_allocated()}),flush=True)
    assert resources()['dedicated_gpu_bytes']>256*2**20,'DXGI did not observe CUDA allocation'
