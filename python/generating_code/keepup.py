#!/software/.admin/bins/bin/python2.7
import sys
import time
import subprocess
from numpy.random import shuffle

print "Starting keepup"

"""
Keep a process up and running

Use it like this:
./keepup.py ./moran.py r2
--or--
./keepup.py ./moran.py u2
"""

if __name__ == "__main__":
    MAX = 50

    r2s = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]
    u2s = [0.1*(1+i) for i in range(9)]

    which_param = sys.argv[-1]

    if which_param == "u2":
        params = u2s
    elif which_param == "r2":
        params = r2s
    else:
        print "Do not recognise the final argument. Should be 'u2' or 'r2'"
        sys.exit(1)

    shuffle(params)

    cmd = ' '.join(sys.argv[1:] + [str(params.pop(0))])

    def start_subprocess():
        return subprocess.Popen(cmd, shell=True)

    p = start_subprocess()
    i = 0

    while True:
        res = p.poll()
        if res is not None:
            if i == MAX - 1: # Counting starts at 0
                i = 0
                cmd = ' '.join(sys.argv[1:] + [str(params.pop(0))])
            else:
                i += 1
            p = start_subprocess()

            print p.pid, 'was killed, restarting it with cmd ' + cmd

        time.sleep(0.5)

