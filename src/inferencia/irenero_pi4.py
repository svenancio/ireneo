# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main script to run the object detection routine."""
import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

from playsound import playsound
import concurrent.futures
import random

#global variables
sound_queue = []
currently_playing = set()
max_simultaneous_sounds = 3

#gerador de threads para abrir cada audio concorrentemente, sem travar o processamento principal
executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_simultaneous_sounds)


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
        

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)
    if len(detection_result.detections) > 0:
       detection_id = detection_result.detections[0].categories[0].index + 1 #soma 1 para corrigir a listagem
       print(detection_result.detections[0].categories[0].index)
       print(detection_result.detections[0].categories[0].score)
       print(detection_result.detections[0].categories[0].category_name)
       submit_sound(f"label-audio/{detection_id}_{random.randint(0,2)}.wav")
    #submit_sound("label-audio/"+str(detection.ClassID)+"_"+str(random.randint(0,2))+".wav")
    # Draw keypoints and edges on input image
    #image = utils.visualize(image, detection_result)

    # Calculate the FPS
    #if counter % fps_avg_frame_count == 0:
      #end_time = time.time()
      #fps = fps_avg_frame_count / (end_time - start_time)
      #start_time = time.time()

    # Show the FPS
    #fps_text = 'FPS = {:.1f}'.format(fps)
    #text_location = (left_margin, row_size)
    #cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                #font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    #if cv2.waitKey(1) == 27:
      #break
    #cv2.imshow('object_detector', image)
    
    time.sleep(0.5) #aguarda meio segundo antes da próxima inferência

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='custom_model_lite/detect_metadata.tflite') #efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()
