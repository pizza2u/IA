import cv2

classificadorCarro = cv2.CascadeClassifier('cascades\\carros.xml') # utiliza um haarcascade treinado para detectar carros

imagem = cv2.imread('outros\\carros1.jpg') # 1.01; # 9; #70,70 - atente-se para o caminho e extensão da imagem
#imagem = cv2.imread('outros\\carros2.jpg') # 1.053; #9
#imagem = cv2.imread('outros\\carros3.jpg') # 1.02; # 8
#imagem = cv2.imread('outros\\carros4.jpg') # 1.01; # 8
#imagem = cv2.imread('outros\\carros5.jpg')  # 1.01; # 9; #70,70

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # transforma a imagem colorida em escala de cinza
# comando para detectar carros na imagem em escala de cinza. Você pode alterar os parâmetros scaleFactor e minNeighbors
# para melhorar a precisão da detecção (alguns estão indicados como os números na frente da imagem acima)

detectado = classificadorCarro.detectMultiScale(imagemCinza, scaleFactor=1.01, minNeighbors=9, minSize=(70, 70))

for (x, y, l, a) in detectado:
    imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2) # comando para desenhar um retângulo na presença de carros

cv2.imshow(str(len(detectado)) + ' carro(s) encontrado(s)', imagem) # mostrará a quantidade de carros no título da janela
cv2.waitKey() # comando que aguarda o fechamento das janelas com as imagens