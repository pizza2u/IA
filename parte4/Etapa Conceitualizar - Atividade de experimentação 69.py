"""
Inteligência Artificial aplicada à Visão Computacional
Capítulo 7: Visão Computacional aplicada ao rastreamento de objetos
-----------------------------------------------------------------------------------------------------------------------
ORIENTAÇÕES:

#1 - Antes de iniciar e executar o código, abra a aba Terminal, localizada na parte inferior do PyCharm e execute, na
sequência, os seguintes comandos para instalar os recursos da biblioteca do OpenCV:

pip install opencv-python

pip install opencv-contrib-python

#2 - Lembre-se de trazer a pasta videos disponibilizada para dentro do PyCharm. Você pode arrastar a pasta para dentro do
projeto, no menu lateral esquerdo.

#3 - Para executar o código:
    * Clique em Run;
    * Ao iniciar a janela do vídeo, selecione com o mouse criando um retângulo no objeto de interesse para o rastreamento;
    * Aperte ESPAÇO para selecionar o objeto;
    * Para selecionar novos objetos, aperte ESPAÇO mais uma vez;
    * Pressione Q para executar o rastreamento;
    * Pressione ESC para encerrar a qualquer momento.

-----------------------------------------------------------------------------------------------------------------------

Atividade de experimentação 69

"""

# Importando bibliotecas
import cv2
import sys
from random import randint

# Declarando os tipos de algoritmos de rastreamento
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']

# Escolhendo o algoritmo 0: BOOSTING, 1: MIL, 2: KCF, 3: TLD, 4: MEDIANFLOW, 5: MOSSE, 6: CSRT
def createTrackerByName(trackerType):
    if trackerType == tracker_types[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif trackerType == tracker_types[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif trackerType == tracker_types[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif trackerType == tracker_types[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif trackerType == tracker_types[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif trackerType == tracker_types[5]:
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif trackerType == tracker_types[6]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('Nome incorreto')
        print('Os rastreadores disponíveis são: ')
        for t in tracker_types:
            print(t)

    return tracker

cap = cv2.VideoCapture("videos/race.mp4") # Localize o caminho do vídeo para análise, na pasta videos.

ok, frame = cap.read()
if not ok:
    print('Não é possível ler o arquivo de vídeo')
    sys.exit(1)

# Cria diferentes retângulos, para diferentes objetos e diferentes cores.
bboxes = []
colors = []

while True:
    bbox = cv2.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0,255), randint(0,255)))
    print('Pressione Q para sair das caixas de seleção e começar a rastrear') # Atente-se às instruções.
    print('Pressione qualquer outra tecla para selecionar o próximo objeto')
    k = cv2.waitKey(0) & 0XFF
    if (k == 113):
        break

print('Caixas delimitadoras selecionadas {}'.format(bboxes))
print('Cores {}'.format(colors))

trackertype = 'CSRT' # Indique o algoritmo a ser utilizado.
multiTracker = cv2.legacy.MultiTracker_create()

# Criando o rastreamento.
for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackertype), frame, bbox)

# Comandos para rastrear o objeto enquando vídeo estiver ativo.
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    ok, boxes = multiTracker.update(frame)

    for i, newbox in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in newbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors[i], 2, 1)

    cv2.imshow('MultiTracker', frame)

    if cv2.waitKey(1) & 0XFF == 27:
        break