#para executar este código, é necessário que o Raspberry Pi Zero 2 W esteja conectado com câmera Raspberry, em estado funcional

from time import sleep
import picamera

#defina em segundos o tempo de espera entre uma foto e outra
WAIT_TIME = 60

#defina o caminho para armazenar as fotos
FOLDER_PATH = "/home/pi/ireneo/img/"

#defina o tamanho da fotografia quadrada em pixels
SIZE = 1080

#estrutura que se repete, tirando fotos e armazenando a data e horário no nome do arquivo
with picamera.PiCamera() as camera:
  camera.resolution = (SIZE, SIZE)
  camera.shutter_speed = camera.exposure_speed
  for filename in camera.capture_continuous(f"{FOLDER_PATH}img{timestamp:%d%m%Y-%H%M%S}.jpg"):
	sleep(WAIT_TIME)
