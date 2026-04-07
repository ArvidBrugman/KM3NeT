#!/usr/bin/env python

import os
import os.path
import sys
from optparse import OptionParser
from ACpulse import ACpulse  
import numpy as np

    

def main(argv):
 
# input parser
    # if user does not provide any input, use defaults
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.set_defaults(Rpos=1000)
    parser.set_defaults(Zpos=12.)
    parser.set_defaults(Edep=1e20)


    # commands to run it like: python pulse.py -R 500 -Z 20 -E 1e19
    parser.add_option("-R", "--rpos", type="float", dest="Rpos",
                      help="radial pos hydrophone (m), default = 1000")
    parser.add_option("-Z", "--zpos", type="float", dest="Zpos",
                      help="z pos hydrophone (m), default = 12")
    parser.add_option("-E", "--energy", type="float", dest="Edep",
                      help="Eenergy deposition, default = 1e20 eV")

    # get the input and set it to parameters
    (options, args) = parser.parse_args()

    Rpos = options.Rpos
    Zpos = options.Zpos
    pos = ([Rpos, Zpos])
    Edep = options.Edep

    # decides how the pulse will be calculated and gets the signal
    pulse = ACpulse()
    pulse.hydrophonePosition(pos)
    pulse.shower_energy(Edep)

    # gives time (x-axis) and amplitude (y-axis) of the pulse
    # outcome is a waveform
    time_axis, signal = pulse.getSignal()

    # takes the maximum value of the signal 
    signal_max = np.amax(signal)

    # generates a file
    # columns: time, signal, normalized signal
    filename = 'pulse_{0}_{1}_{2}.dat'.format(Edep, int(pos[0]), int(pos[1]))
    np.savetxt(filename, np.transpose([time_axis, signal, signal/signal_max]))

    print("maximum signal    ", signal_max)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
