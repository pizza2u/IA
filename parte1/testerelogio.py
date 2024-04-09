import cv2

classificadorRelogio = cv2.CascadeClassifier('cascades\\relogios.xml') # utiliza um haarcascade treinado para detectar relógios

imagem = cv2.imread('outros\\relogios1.jpg')  # 1.07 - atente-se para o caminho e extensão da imagem
#imagem = cv2.imread('outros\\relogios2.jpg') # 1.068; 6
#imagem = cv2.imread('outros\\relogios3.jpg') # 1.068
#imagem = cv2.imread('outros\\relogios4.jpg') # 1.05
#imagem = cv2.imread('outros\\relogios5.jpg') # 1.05
#imagem = cv2.imread('outros\\relogios6.jpg') # 1.05

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # transforma a imagem colorida em escala de cinza

detectado = classificadorRelogio.detectMultiScale(imagemCinza, scaleFactor=1.07, minNeighbors=6)
# comando para detectar relógios na imagem em escala de cinza. Você pode alterar os parâmetros scaleFactor e minNeighbors
# para melhorar a precisão da detecção (alguns estão indicados como os números na frente da imagem acima)

for (x, y, l, a) in detectado:
    imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2) # comando para desenhar um retângulo na presença de relógio

cv2.imshow(str(len(detectado)) + ' relogio(s) encontrado(s)', imagem) # mostrará a quantidade de relógios no título da janela
cv2.waitKey() # comando que aguarda o fechamento das janelas com as imagens