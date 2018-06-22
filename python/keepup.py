#!/software/.admin/bins/bin/python2.7

print "Starting keepup"

import sys
import time
import subprocess

"""
Keep a process up and running

If you have a long running process that can be killed for strange and unknown
reason, you might want it to be restarted ... this script does that.

$ cat alive.sh 
#!/bin/sh

while `true`; do echo Alive && sleep 3 ; done

Use it like this:
$ keepup.py ./alive.sh 
"""

MAX = 50

u2s = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]

cmd = ' '.join(sys.argv[1:] + [str(u2s.pop(0))])

def start_subprocess():
    return subprocess.Popen(cmd, shell=True)

p = start_subprocess()
i = 0

while True:
    res = p.poll()
    if res is not None:
        if i == MAX - 1: # Counting starts at 0
            i = 0
            cmd = ' '.join(sys.argv[1:] + [str(u2s.pop(0))])
        else:
            i += 1
        p = start_subprocess()

        print p.pid, 'was killed, restarting it with cmd ' + cmd

    time.sleep(0.5)
