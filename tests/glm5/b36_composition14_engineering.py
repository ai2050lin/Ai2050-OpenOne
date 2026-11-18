"""Protocol regression with task-local storage and periodic flushed progress."""
import io,os,sys,tempfile,threading,time,unittest
import b36_composition14 as B


def run():
    B.configure(200*2**20)
    folder=B.TEMP/'protocol';folder.mkdir(exist_ok=True)
    for k in ['TEMP','TMP','TMPDIR']:os.environ[k]=str(folder)
    tempfile.tempdir=str(folder)
    fixture=folder/'research-fixture';fixture.mkdir(exist_ok=True)
    os.environ['AI2050_RESEARCH_DIR']=str(fixture)
    start=time.monotonic();stop=threading.Event();progress={'done':0,'total':0,'current':None}
    class Result(unittest.TextTestResult):
        def startTest(self,test):
            super().startTest(test);progress['current']=str(test)
        def stopTest(self,test):
            super().stopTest(test);progress['done']=self.testsRun
    suite=unittest.defaultTestLoader.discover(str(B.ROOT/'tests/dist_protocol'))
    progress['total']=suite.countTestCases()
    def heartbeat():
        while not stop.wait(30):
            try:B.log('protocol_regression',progress['done'],progress['total'],start,current=progress['current'])
            except Exception as error:
                print('Regression stopped by resource monitor:',type(error).__name__,str(error),flush=True)
                os._exit(3)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    B.log('protocol_regression',0,progress['total'],start)
    try:result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    finally:stop.set();thread.join(1)
    B.save('protocol-regression.json',dict(kind='engineering_only',tests=result.testsRun,failures=[dict(test=str(t),traceback=s) for t,s in result.failures],
        errors=[dict(test=str(t),traceback=s) for t,s in result.errors],skipped=[dict(test=str(t),reason=s) for t,s in result.skipped],
        passed=result.wasSuccessful(),source_sha256=B.file_sha(__file__),seconds=time.monotonic()-start,
        isolated_research_directory=str(fixture),fixture_scope='All315 unchanged tests. Documented AI2050_RESEARCH_DIR redirects source library scan away from unrelated real research; no production code patched.'))
    B.log('protocol_regression',result.testsRun,progress['total'],start)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':sys.exit(run())
