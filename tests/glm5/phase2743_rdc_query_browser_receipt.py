"""Record live-browser initialization errors without claiming UI control."""
from rdc_query_common import *

def main():
    immutable(BASE/'client/live_browser_attempt.json',{'timestamp':stamp(),'source':snapshot(__file__),
      'attempts':['cua.getState()','JavaScriptkernelreset then cua.getState()'],
      'both_errors':'failed to write kernel assets: 系统找不到指定的路径。 (os error 3)',
      'actual_user_browser_controlled':False,'fallback':'Independent ephemeral headless Edge application tests, no user profile or active tab attached.',
      'skill':'computer-use SKILL, guidance and confirmations read completely; browser-use entry preferred for browser tasks.'})
    print('LIVE_BROWSER_LIMIT_RECORDED')

if __name__=='__main__':main()
