import cv2

classificadorGato = cv2.CascadeClassifier('cascades\\gatos.xml') # utiliza um haarcascade treinado para detectar gatos

imagem = cv2.imread('outros\\gatos1.jpg')  # 1.03; #10 - atente-se para o caminho e extensão da imagem
#imagem = cv2.imread('outros\\gatos2.jpg') # 1.2; # 2
#imagem = cv2.imread('outros\\gatos3.jpg') #1.02; 9
#imagem = cv2.imread('outros\\gatos4.jpg') # 1.08; #10
#imagem = cv2.imread('outros\\gatos5.jpg') # 1.069; #10

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # transforma a imagem colorida em escala de cinza

detectado = classificadorGato.detectMultiScale(imagemCinza, scaleFactor=1.02, minNeighbors=9)
# comando para detectar gatos na imagem em escala de cinza. Você pode alterar os parâmetros scaleFactor e minNeighbors
# para melhorar a precisão da detecção (alguns estão indicados como os números na frente da imagem acima)

for (x, y, l, a) in detectado:
    imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2) # comando para desenhar um retângulo na presença de gatos

cv2.imshow(str(len(detectado)) + ' gato(s) encontrado(s)', imagem) # mostrará a quantidade de gatos no título da janela
cv2.waitKey() # comando que aguarda o fechamento das janelas com as imagens