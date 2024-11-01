#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log

from playsound import playsound
import concurrent.futures
import time
import random


debugging = True #use this to debug general things

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

#is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# create video sources and outputs
input = videoSource(args.input, argv=sys.argv)
#output = videoOutput(args.output, argv=sys.argv+is_headless)
	
# load the object detection network
net = detectNet(args.network, sys.argv, args.threshold)

# note: to hard-code the paths to load a model, the following API can be used:
#
# net = detectNet(model="model/ssd-mobilenet.onnx", labels="model/labels.txt", 
#                 input_blob="input_0", output_cvg="scores", output_bbox="boxes", 
#                 threshold=args.threshold)

sound_queue = []
currently_playing = set()
max_simultaneous_sounds = 3

def play_sound(sound_name):
    playsound(sound_name)
    currently_playing.remove(sound_name)
    play_next_queued_sound()

def play_next_queued_sound():
    if sound_queue:
        next_sound = sound_queue.pop(0)
        print(f"Playing {next}")
        currently_playing.add(next_sound)
        executor.submit(play_sound, next_sound)

def submit_sound(sound_name):
    if sound_name in currently_playing:
        print(f"{sound_name} is already playing. Queueing it.")
        sound_queue.append(sound_name)
    elif len(currently_playing) < max_simultaneous_sounds:
        print(f"Playing {sound_name}")
        currently_playing.add(sound_name)
        executor.submit(play_sound, sound_name)
    else:
        print(f"{sound_name} is queued")


#gerador de threads para abrir cada audio concorrentemente, sem travar o processamento principal
executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_simultaneous_sounds)

if debugging:
    try:
        for i in range(0,91):
            print(f"{i}: {net.GetClassLabel(i)}")
    except:
        print(f"{i} deu erro")

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  
        
    # detect objects in the image (with overlay)
    detections = net.Detect(img, overlay='none')

    # print the detections
    #print("detected {:d} objects in image".format(len(detections)))

    for detection in detections:
        #print(detection)
        print(net.GetClassLabel(detection.ClassID) + " | " + str(detection.Confidence) +"%")
        submit_sound("label-audio/"+str(detection.ClassID)+"_"+str(random.randint(0,2))+".wav")

    # render the image
    #output.Render(img)

    # update the title bar
    #output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    #net.PrintProfilerTimes()

    time.sleep(0.5) #aguarda meio segundo antes da próxima inferência

    # exit on input/output EOS
    #if not input.IsStreaming() or not output.IsStreaming():
    if not input.IsStreaming():
        break
