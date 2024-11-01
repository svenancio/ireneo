import argparse
import sys
import time

"""
importação de bibliotecas OpenCV e TFLite, para processamento do modelo treinado 
e produção de imagens com visualização de bounding boxes
"""
import cv2 
from tflite_support.task import core 
from tflite_support.task import processor
from tflite_support.task import vision
import utils

from playsound import playsound
import concurrent.futures
import random

"""
  DEBUG: coloque True para que o código produza resultados visuais do processo de detecção (precisa estar conectado em monitor)
  Mantenha False quando não estiver conectado a um monitor
"""
DEBUG = False

#estruturas para fila de reprodução sonora
sound_queue = []
currently_playing = set()
max_simultaneous_sounds = 3

#gerador de threads para abrir cada som concorrentemente, sem travar o processamento principal
executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_simultaneous_sounds)

#função para execução do som
def play_sound(sound_name):
    playsound(sound_name)
    currently_playing.remove(sound_name)
    play_next_queued_sound()

#função para fazer a fila de espera de sons andar
def play_next_queued_sound():
    if sound_queue:
        next_sound = sound_queue.pop(0)
        print(f"Tocando {next_sound}")
        currently_playing.add(next_sound)
        executor.submit(play_sound, next_sound)

#enfileira os sons a serem reproduzidos
def submit_sound(sound_name):
    if sound_name in currently_playing:
        print(f"{sound_name} já está tocando. Enfileirando.")
        sound_queue.append(sound_name)
    elif len(currently_playing) < max_simultaneous_sounds:
        print(f"Tocando {sound_name}")
        currently_playing.add(sound_name)
        executor.submit(play_sound, sound_name)
    else:
        print(f"{sound_name} está enfileirado")

#executa continuamente a tarefa de detecção de objetos em imagens de uma camera
def run(model: str, camera_id: int, width: int, height: int, num_threads: int) -> None:
  """
    Args:
    model: Nome do modelo TFLite
    camera_id: ID da câmera a ser passada para a OpenCV
    width: largura da imagem a ser capturada pela câmera
    height: altura da imagem a ser capturada pela câmera
    num_threads: Número de threads de CPU para executar o modelo
  """

  # variáveis para cálculo de FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Inicia a captura de vídeo a partir da câmera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Parâmetros de visualização (necessários apenas em modo debug)
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Inicializa o modelo de detecção
  base_options = core.BaseOptions(file_name=model, use_coral=False, num_threads=num_threads)
  detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.5)
  options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # execução contínua das detecções a partir de imagens da câmera, com intervalo definido
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERRO: Impossível ler a câmera. Verifique sua conexão e configuração.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Converte a imagem de BGR para RGB, como solicitado pelo modelo TFLite
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # cria um objeto TensorImage a partir da imagem
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Executa a detecção sobre o TensorImage usando o modelo
    detection_result = detector.detect(input_tensor)

    # se houver detecção com sucesso
    if len(detection_result.detections) > 0:
      #recupera uma detecção aleatória dentre as possíveis
      detection_id = detection_result.detections[0].categories[0].index + 1 #soma 1 para corrigir a listagem
       
      print(detection_result.detections[0].categories[0].index)
      print(detection_result.detections[0].categories[0].score)
      print(detection_result.detections[0].categories[0].category_name)

      #TODO: corrigir a quantidade de memórias por rótulo
      submit_sound(f"label-audio/{detection_id}_{random.randint(0,2)}.wav")

      if DEBUG:
        # Desenha a bounding box rotulada sobre a imagem
        image = utils.visualize(image, detection_result)

        # Calcula o FPS
        if counter % fps_avg_frame_count == 0:
          end_time = time.time()
          fps = fps_avg_frame_count / (end_time - start_time)
          start_time = time.time()

        # Exibe o FPS
        fps_text = 'FPS = {:.1f}'.format(fps)
        text_location = (left_margin, row_size)
        cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)

        cv2.imshow('object_detector', image)
    
    # Para o programa se a tecla ESC for pressionada
    if cv2.waitKey(1) == 27:
      break

    time.sleep(1.0) #aguarda um segundo antes da próxima inferência

  cap.release()
  cv2.destroyAllWindows()


# a função main faz o tratamentos dos parâmetros invocados na inicialização
def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='custom_model_lite/detect_metadata.tflite')
  
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
    
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=320)
    
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=320)
    
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
    
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight, int(args.numThreads))

if __name__ == '__main__':
  main()
