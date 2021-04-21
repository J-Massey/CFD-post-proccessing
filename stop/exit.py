# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: Graceful stopping condition
@contact: jmom1n15@soton.ac.uk
"""

import signal
import time
import subprocess
from pathlib import Path

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self,signum, frame):
        self.kill_now = True    

if __name__ == '__main__':
    try:
        subprocess.call('rm .kill', shell=True, cwd=Path.cwd())
    except FileNotFoundError:
        print("No .kill file present. You are cleared for takeoff")
    killer = GracefulKiller()
    while not killer.kill_now:  
        time.sleep(1)
    print("Found SIGINT/SIGTERM signal")
    subprocess.call('touch .kill', shell=True, cwd=Path.cwd())
    print("Clean Exit. The flow field has been saved :)")