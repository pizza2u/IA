import cv2

classificador = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml') # utiliza um haarcascade treinado para detectar faces

imagem = cv2.imread('pessoas\\beatles.jpg')  #1.15, #7, #20x20 - atente-se para o caminho e extensão da imagem
#imagem = cv2.imread('pessoas\\pessoas1.jpg') #1.08 , #6
#imagem = cv2.imread('pessoas\\pessoas2.jpg') #1.2 // #1.15, #7
#imagem = cv2.imread('pessoas\\pessoas3.jpg') #
#imagem = cv2.imread('pessoas\\pessoas4.jpg') #1.01 // #1.01, #9
#imagem = cv2.imread('pessoas\\pessoas5.jpg')
#imagem = cv2.imread('pessoas\\pessoas6.jpg')
#imagem = cv2.imread('pessoas\\pessoas7.jpg')
#imagem = cv2.imread('pessoas\\faceolho.jpg')
#imagem = cv2.imread('pessoas\\olho.jpg')

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # transforma a imagem colorida em escala de cinza

facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.15, minNeighbors=7, minSize=(20, 20))
# comando para detectar faces na imagem em escala de cinza. Você pode alterar os parâmetros scaleFactor, minNeighbors e
# minSize para melhorar a precisão da detecção (alguns estão indicados como os números na frente da imagem acima)

for (x, y, l, a) in facesDetectadas:
    #print(x, y, l, a)
    imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2) # comando para desenhar um retângulo na presença de faces

cv2.imshow(str(len(facesDetectadas)) + " face(s) encontrada(s)", imagem) # mostrará a quantidade de faces no título da janela
cv2.waitKey() # comando que aguarda o fechamento das janelas com as imagens