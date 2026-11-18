"""Bounded serial runner; only GPU lease contention is retried."""
import argparse,time
import b36_composition14 as B


def run(stage,split):
    B.configure();start=time.monotonic()
    while True:
        try:
            if stage=='collect':
                from b36_composition14_collect import collect
                collect(split)
            elif stage=='tail-pilot':
                from b36_composition14_tail import run as tail
                tail('pilot',True)
            elif stage=='tail':
                from b36_composition14_tail import run as tail
                tail(split)
            elif stage=='output':
                from b36_composition14_output import run as output
                output(split)
            break
        except RuntimeError as error:
            if str(error)!='GPU is leased by another research/live process' or time.monotonic()-start>3600:raise
            B.log('waiting_'+stage+'_'+split,0,1,start);time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','tail-pilot','tail','output']);p.add_argument('--split',default='pilot');a=p.parse_args();run(a.stage,a.split)
