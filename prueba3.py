import numpy as np
import cv2
import pandas as pd

cap = cv2.VideoCapture("/Users/aaronsantiagopedrazacardenas/Desktop/procesamiento de video /proyecto/videos proyecto/77.3gp")

frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = int(width)
height = int(height)
print(frames_count, fps, width, height)

# crear un marco de datos de pandas con el número de filas de la misma longitud que el recuento de marcos
df = pd.DataFrame(index=range(int(frames_count)))
df.index.name = "Frames"

framenumber = 0  # realiza un seguimiento del cuadro actual
carscrossedup = 0  # realiza un seguimiento de los coches que cruzaron
carscrosseddown = 0  # keeps track of cars that crossed down
carids = []  # lista en blanco para agregar identificadores de automóviles
caridscrossed = []  # lista en blanco para agregar identificadores de automóviles que se han cruzado
totalcars = 0  # realiza un seguimiento del total de autos

fgbg = cv2.createBackgroundSubtractorMOG2()  # crear un restador de fondo

# information to start saving a video file
ret, frame = cap.read()  # importar el video
ratio = .5  # relacion del resize
image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize del video
width2, height2, channels = image.shape
video = cv2.VideoWriter('traffic_counter.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2), 1)

while True:

    ret, frame = cap.read()  # import image

    if ret:  # if there is a frame continue with code

        image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray

        fgmask = fgbg.apply(gray)  # uses the background subtraction

        # applies different thresholds to fgmask to try and isolate cars
        # just have to keep playing around with settings until cars are easily identifiable
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(opening, kernel)
        retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows

        # creates contours
        contours, hierarchy = cv2.findContours(bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # use convex hull to create polygon around contours
        hull = [cv2.convexHull(c) for c in contours]

        # draw contours
        cv2.drawContours(image, hull, -1, (0, 255, 0), 3)

        # line created to stop counting contours, needed as cars in distance become one big contour
        #lineypos = 390
        lineypos = 170
        cv2.line(image, (0, lineypos), (width, lineypos), (255, 0, 0), 5)

        # line y position created to count contours
        #lineypos2 = 5
        lineypos2 = 400
        cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 255, 0), 5)

        # min area for contours in case a bunch of small noise contours are created
        minarea = 600

        # max area for contours, can be quite large for buses
        maxarea = 50000

        # vectores para las ubicaciones X y Y de los centroides de contorno en el cuadro actual
        cxx = np.zeros(len(contours))
        cyy = np.zeros(len(contours))

        for i in range(len(contours)):  # recorre todos los contornos en el cuadro actual

            if hierarchy[0, i, 3] == -1:  # usar la jerarquía para contar solo los contornos principales (los contornos no están dentro de otros)

                area = cv2.contourArea(contours[i])  # area del contorno

                if minarea < area < maxarea:  # umbral de área para contorno

                    # calcular centroides de contornos
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    if cy > lineypos:  # iltra los contornos que están por encima de la línea (y comienza en la parte superior)

                        # obtiene puntos delimitadores del contorno para crear un rectángulo
                        # x, y es la esquina superior izquierda y w, h es ancho y alto
                        x, y, w, h = cv2.boundingRect(cnt)

                        # crea un rectángulo alrededor del contorno
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        # Imprime el texto del centroide para verificarlo más adelante
                        cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    .3, (0, 0, 255), 1)

                        cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                       line_type=cv2.LINE_AA)

                        # agrega centroides que pasaron los criterios anteriores a la lista de centroides
                        cxx[i] = cx
                        cyy[i] = cy

        # elimina cero entradas (centroides que no se agregaron)
        cxx = cxx[cxx != 0]
        cyy = cyy[cyy != 0]

        # lista vacía para verificar más tarde qué índices de centroide se agregaron al marco de datos
        minx_index2 = []
        miny_index2 = []

        # radio máximo permitido para que el centroide del marco actual se considere el mismo centroide del marco anterior
        maxrad = 25

        # La siguiente sección realiza un seguimiento de los centroides y los asignaa a old carids o new carids

        if len(cxx):  # si hay centroides en el área especificada

            if not carids:  # Si carids está vacía

                for i in range(len(cxx)):  # recorre todos los centroides

                    carids.append(i)  # agrega un automóvil a las tarjetas de lista vacías
                    df[str(carids[i])] = ""  # agrega una columna al marco de datos correspondiente a un carid

                    # asigna los valores de centroide al marco actual (fila) y carid (columna)
                    df.at[int(framenumber), str(carids[i])] = [cxx[i], cyy[i]]

                    totalcars = carids[i] + 1  # agrega un recuento al total de autos

            else:  # si ya hay identificaciones de autos

                dx = np.zeros((len(cxx), len(carids)))  # nuevas matrices para calcular deltas
                dy = np.zeros((len(cyy), len(carids)))  # nuevas matrices para calcular deltas

                for i in range(len(cxx)):  # recorre todos los centroides

                    for j in range(len(carids)):  # recorre todos los identificadores de automóviles registrados

                        # adquiere el centroide del fotograma anterior para carid específico
                        oldcxcy = df.iloc[int(framenumber - 1)][str(carids[j])]

                        # adquiere el centroide del marco actual que no necesariamente se alinea con el centroide del marco anterior
                        curcxcy = np.array([cxx[i], cyy[i]])

                        if not oldcxcy:  # comprueba si el centroide antiguo está vacío en caso de que el coche deje la pantalla y se muestre el coche nuevo

                            continue  # continuar con la siguiente carid

                        else:  # calcular los deltas del centroide para compararlos con la posición actual del cuadro más adelante

                            dx[i, j] = oldcxcy[0] - curcxcy[0]
                            dy[i, j] = oldcxcy[1] - curcxcy[1]

                for j in range(len(carids)):  # recorre todos los identificadores de automóviles actuales

                    sumsum = np.abs(dx[:, j]) + np.abs(dy[:, j])  # suma los deltas wrt a los identificadores de automóviles

                    # encuentra qué índice carid tuvo la diferencia mínima y este es el índice verdadero
                    correctindextrue = np.argmin(np.abs(sumsum))
                    minx_index = correctindextrue
                    miny_index = correctindextrue

                    # adquiere valores delta de los deltas mínimos para verificar si está dentro del radio más adelante
                    mindx = dx[minx_index, j]
                    mindy = dy[miny_index, j]

                    if mindx == 0 and mindy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                        # comprueba si el valor mínimo es 0 y comprueba si todos los deltas son cero, ya que es un conjunto vacío
                        # La delta podría ser cero si el centroide no se movió

                        continue  # continuar con la siguiente carid
                    else:

                        # si los valores delta son menores que el radio máximo, agregue ese centroide a ese carid específico
                        if np.abs(mindx) < maxrad and np.abs(mindy) < maxrad:

                            # agrega centroide a la carid correspondiente previamente existente
                            df.at[int(framenumber), str(carids[j])] = [cxx[minx_index], cyy[miny_index]]
                            minx_index2.append(minx_index)  # agrega todos los índices que se agregaron a los carids anteriores
                            miny_index2.append(miny_index)

                for i in range(len(cxx)):  # recorre todos los centroides

                    #si el centroide no está en la lista del miníndice, entonces se debe agregar otro automóvil
                    if i not in minx_index2 and miny_index2:

                        df[str(totalcars)] = ""  # crea otra columna con autos totales
                        totalcars = totalcars + 1  # agrega otro carro total el recuento
                        t = totalcars - 1  # t es un marcador de posición para autos totales
                        carids.append(t)  # agregar a la lista de identificadores de automóviles
                        df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # agregar centroide a la nueva identificación del automóvil

                    elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                        # comprueba si el centroide actual existe pero el centroide anterior no
                        # Se agregará un nuevo automóvil en caso de que minx_index2 esté vacío

                        df[str(totalcars)] = ""  # crea otra columna con autos totales
                        totalcars = totalcars + 1  # agrega otro carro total el recuento
                        t = totalcars - 1  # t es un marcador de posición para autos totales
                        carids.append(t)  # agrega la lista de car ids
                        df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # suma el centroide del new car id

        # La siguiente sección etiqueta los centroides en la pantalla

        currentcars = 0  # coches actuales en pantalla
        currentcarsindex = []  # coches actuales en la pantalla índice carid

        for i in range(len(carids)):  # recorre todo carids

            if df.at[int(framenumber), str(carids[i])] != '':
                #  comprueba el marco actual para ver qué ID de coche están activos
                #al verificar que el centroide existe en el marco actual para cierta identificación de automóvil

                currentcars = currentcars + 1  #agrega otro a los autos actuales en pantalla
                currentcarsindex.append(i)  # agrega identificadores de automóviles a los automóviles actuales en la pantalla

        for i in range(currentcars):  #recorre todos los identificadores de automóviles actuales en la pantalla

            # toma el centroide de cierto carid para el marco actual
            curcent = df.iloc[int(framenumber)][str(carids[currentcarsindex[i]])]

            # toma el centroide de cierto carid para el cuadro anterior
            oldcent = df.iloc[int(framenumber - 1)][str(carids[currentcarsindex[i]])]

            if curcent:  # si hay un centroide actual

                # Texto en pantalla para el centroide actual
                cv2.putText(image, "Centroid" + str(curcent[0]) + "," + str(curcent[1]),
                            (int(curcent[0]), int(curcent[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                cv2.putText(image, "ID:" + str(carids[currentcarsindex[i]]), (int(curcent[0]), int(curcent[1] - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                cv2.drawMarker(image, (int(curcent[0]), int(curcent[1])), (0, 0, 255), cv2.MARKER_STAR, markerSize=5,
                               thickness=1, line_type=cv2.LINE_AA)

                if oldcent:  # comprueba si existe un centroide antiguo
                    # agrega el cuadro de radio del centroide anterior al centroide actual para la visualización
                    xstart = oldcent[0] - maxrad
                    ystart = oldcent[1] - maxrad
                    xwidth = oldcent[0] + maxrad
                    yheight = oldcent[1] + maxrad
                    cv2.rectangle(image, (int(xstart), int(ystart)), (int(xwidth), int(yheight)), (0, 125, 0), 1)

                    # comprueba si el centroide antiguo está en la línea o debajo de ella y si la curva está en la línea o arriba
                    # para contar automóviles y ese automóvil aún no se ha contado
                    if oldcent[1] >= lineypos2 and curcent[1] <= lineypos2 and carids[
                        currentcarsindex[i]] not in caridscrossed:

                        carscrossedup = carscrossedup + 1
                        cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 0, 255), 5)
                        caridscrossed.append(
                            currentcarsindex[i])  # agrega la identificación del automóvil a la lista de automóviles de recuento para evitar el conteo doble

                    # comprueba si el centroide antiguo está en la línea o por encima de ella y si la curva está en la línea o debajo de ella
                    # para contar automóviles y ese automóvil aún no se ha contado
                    elif oldcent[1] <= lineypos2 and curcent[1] >= lineypos2 and carids[
                        currentcarsindex[i]] not in caridscrossed:

                        carscrosseddown = carscrosseddown + 1
                        cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 0, 125), 5)
                        caridscrossed.append(currentcarsindex[i])

        # TTexto en pantalla de la esquina superior izquierda
        cv2.rectangle(image, (0, 0), (250, 100), (255, 0, 0), -1)  # rectángulo de fondo para texto en pantalla

        cv2.putText(image, "Cars in Area: " + str(currentcars), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)

        cv2.putText(image, "Cars Crossed Up: " + str(carscrossedup), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0),
                    1)

        cv2.putText(image, "Cars Crossed Down: " + str(carscrosseddown), (0, 45), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (0, 170, 0), 1)

        cv2.putText(image, "Total Cars Detected: " + str(len(carids)), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (0, 170, 0), 1)

        cv2.putText(image, "Frame: " + str(framenumber) + ' of ' + str(frames_count), (0, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 170, 0), 1)

        cv2.putText(image, 'Time: ' + str(round(framenumber / fps, 2)) + ' sec of ' + str(round(frames_count / fps, 2))
                    + ' sec', (0, 90), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)

        # muestra imágenes y transformaciones
        cv2.imshow("countours", image)
        cv2.moveWindow("countours", 0, 0)

        cv2.imshow("fgmask", fgmask)
        cv2.moveWindow("fgmask", int(width * ratio), 0)

        cv2.imshow("closing", closing)
        cv2.moveWindow("closing", width, 0)

        cv2.imshow("opening", opening)
        cv2.moveWindow("opening", 0, int(height * ratio))

        cv2.imshow("dilation", dilation)
        cv2.moveWindow("dilation", int(width * ratio), int(height * ratio))

        cv2.imshow("binary", bins)
        cv2.moveWindow("binary", width, int(height * ratio))

        video.write(image)  # guardar la imagen actual en el archivo de video de antes

        # se suma al recuento de cuadros
        framenumber = framenumber + 1

        k = cv2.waitKey(int(1000/fps)) & 0xff  # int(1000/fps) velocidad normal ya que la tecla de espera está en ms
        if k == 27:
            break

    else:  # si el video está terminado, interrumpa el ciclo

        break

cap.release()
cv2.destroyAllWindows()

# guarda el marco de datos en un archivo csv para su posterior análisis
df.to_csv('traffic.csv', sep=',')