import pandas as pd
import netCDF4 as nc
import numpy as np
import cv2
import os
import errno
import geopandas as gpd
import time
from pyproj import Geod
from shapely.geometry import shape, Polygon, Point, MultiPoint, box


def arco(r1,r2,r3,r4,c):
    ang = [np.arange(0,np.pi/2,0.01), np.arange(np.pi/2,np.pi,0.01),
           np.arange(np.pi,(3/2)*np.pi,0.01),np.arange((3/2)*np.pi,2*np.pi,0.01)]
    grafx, grafy = [], []
    for i,r in enumerate([r1,r2,r3,r4]):
        grafx.append(r*np.cos(ang[i])+c[0])
        grafy.append(r*np.sin(ang[i])+c[1])
    return (np.concatenate(grafx, axis = 0),np.concatenate(grafy, axis = 0))
def bina(imagen,umbral):
    """ Función para binarizar la imagen en función de un umbral determinado"""
    minimo = np.min(imagen)
    indx = np.where(imagen<umbral)
    imagen[indx] = 0
    ind = np.where(imagen == minimo)
    imagen[ind] = 0
    indi = np.where(imagen != 0)
    imagen[indi] = 255
    return imagen
def calculo_radios(i,data,data_base):
    """ Función para determinar los radios por cuadrantes. La distancia desde el centro de
    la tormenta hasta el punto más lejano de cada cuadrante"""
    ctpp = [data['lon'][i],data['lat'][i]]
    if ctpp[1] <= 35:
        ctp = Point(ctpp)
        ctp = gdf_conver(ctp)
        points2 = data_base[data_base['ID']==i]
        if points2['geometry'].empty:
            ROC_NW,ROC_NE,ROC_SW,ROC_SE = -9999,-9999,-9999,-9999
        elif points2['geometry'].iloc[0] is None:
            ROC_NW,ROC_NE,ROC_SW,ROC_SE = -9999,-9999,-9999,-9999
        else:
            points2 = gdf_explode(points2)
            points2.geometry = points2.geometry.apply(multipoint, args=(points2,))
            pnw = gdf_explode(inters(points2,gdf_conver(box(ctpp[0],ctpp[1],-130,40))))
            pne = gdf_explode(inters(points2,gdf_conver(box(ctpp[0],ctpp[1],-10,40))))
            psw = gdf_explode(inters(points2,gdf_conver(box(ctpp[0],ctpp[1],-130,0))))
            pse = gdf_explode(inters(points2,gdf_conver(box(ctpp[0],ctpp[1],-10,0))))
            ROC_NW,ROC_NE,ROC_SW,ROC_SE = dist(pnw,ctp),dist(pne,ctp),dist(psw,ctp),dist(pse,ctp)
    else:
        ROC_NW,ROC_NE,ROC_SW,ROC_SE = -9999,-9999,-9999,-9999
    return ROC_NW,ROC_NE,ROC_SW,ROC_SE
def cambiar_sufijos(df,col):
    lon1 = df[col].to_numpy()
    for i in range(len(lon1)):
        if lon1[i][-1] == 'W':
            lon1[i]  = lon1[i].replace("W","")
            lon1[i] = float(lon1[i]) * -1
        else:
            lon1[i]  = lon1[i].replace("E","").replace("N","")
            lon1[i] = float(lon1[i])
    return lon1
def circulo(c,r):
    """Función para construir un circulo. Los argumentos para la función son:
    C = centro de la circunferencia
    r = medida de la circunferencia
    Ambas medidas deben estar en posición en grados"""
    ang = np.arange(0,np.pi*2,0.01)
    x = r*np.cos(ang)+c[0]
    y = r*np.sin(ang)+c[1]
    return (x,y)
def correc(pic1):
    """Corrección de la imagen en lugares donde no se tiene información"""
    correc = np.min(pic1[np.where(pic1!=np.min(pic1))])
    ind = np.where(pic1== np.min(pic1))
    pic1[ind] = correc-1
    return pic1
def contornos(seg):
    """Función para obtener los contornos de una imagen. Los parametros de entrada
    solo requieren un arreglo matricial para una figura"""
    (contornos,_) = cv2.findContours(np.uint8(seg), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    long = []
    for ii in contornos:
        long.append(len(ii))

    long= np.array(long)
    indx = np.where(long>20)
    contorN = []

    for ij in indx[0]:
        contorN.append(contornos[ij])

    #ARREGLO MATRICIAL DE LOS CONTONRNOS
    nubM = []
    for k in range(len(contorN)):

        N = len(contorN[k])
        L = np.zeros((N,2 ))
        for ik in range(N):
            L[ik,:]= contorN[k][ik][0]
        nubM.append(L)
    if len(nubM) == 1:
        return nubM
    else:
        return np.array(nubM,dtype=object)
def convcoor(nube,dlat= 0.099999994, dlon = 0.1000061):
    """ Función para determinar las coordenadas geograficas de los contornos
    calculados en el programa [contornos]. Los parametros de entrada son:
    nube = la cual representa cada segmento de contornos
    dlat, dlon = son medidas para redefinir los saltos de malla"""
    lonc1 = np.round(np.transpose(((dlon*nube[:,0])-130)),2)
    latc1 = np.round((dlat*nube[:,1]),2)
    coordenadas = list(zip(lonc1,latc1))
    return coordenadas
def crear_carp(direccion):
    try:
        os.mkdir(direccion)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
def data_lec(direccion,datos):
    names = ['date', 'time', 'lat', 'lon', 'MWS', 'CPSL', 'ERMWS', 'R34', 'R50', 'R64', 'R100', 'TCOR']
    df1 = pd.read_csv(direccion+'/'+datos, sep=",", skip_blank_lines=True, header = None, names = names)
    fecha = pd.to_datetime(df1['date'], format='%Y%m%d').apply(lambda x: pd.Series([x.year,x.month,x.day], index = ['yy', 'mm', 'dd']))
    dd = fecha['dd']
    mm = fecha['mm']
    yy = fecha['yy']
    hh = (df1['time']/100).astype(int)
    lat = cambiar_sufijos(df1,'lat')
    lon = cambiar_sufijos(df1,'lon')
    R34 = df1['R34']
    ROUT = df1['TCOR']
    d = {'dd':dd,'mm':mm,'yy':yy,'hh':hh,'lat':lat,'lon':lon,'R34':R34,'R':ROUT}
    data = pd.DataFrame(data=d)
    return data
def data_text(x):
    if x == -9999:
        x = str(int(x))
    else:
        x = "{:.2f}".format(x)
    return x
def dist(psel,center):
    center = center.to_crs(epsg=3395)
    if psel.empty:
        maxi = 0
    else:
        psel = psel.to_crs(epsg=3395)
        dist = psel.geometry.apply(lambda g: center.distance(g)).max()
        maxi = np.round(dist.to_numpy()/1000,2)
        maxi = maxi[0]
    return maxi
def filtcloud(nubes,polyar):
    """ Función para seleccionar las nubes que intersectan el area del ROUT
    calculado con el perfil de viento de W06 en Perez-Alarcon et al (2021)
    Los parametros de entrada son las nubes y el poligono del area del ROUT"""
    if len(nubes) == 1:
        selcon = pd.DataFrame([nubes], columns=['geometry'],dtype=object)
    else:
        selcon = pd.DataFrame(nubes, columns=['geometry'],dtype=object)

    selcon['geometry'] = selcon['geometry'].apply(convcoor).apply(Polygon)
    selcon = gpd.GeoDataFrame(selcon,geometry='geometry',crs='EPSG:4326')
    selcon['geometry'] = selcon.buffer(0)
    indx = selcon['geometry'].apply(lambda nube: nube.intersects(polyar))
    return selcon[indx]
def gdf_conver(gdf):
    """ Función para convertir una serie de datos al crs EPSG:4326, que
    representa una proyección geografica WGS84"""
    return gpd.GeoDataFrame(index=[0], crs='EPSG:4326', geometry=[gdf])
def gdf_explode(gdf):
    """ Separación (explode) de una GeoDataFrame en columnas"""
    return gdf.explode().reset_index(drop=True).rename(columns={0: 'geometry'})
def inters(data,sec):
    if data.empty:
        return data
    else:
        return gpd.overlay(data,sec, how='intersection')
def lec_imag(data,i):
    yy =  data.yy
    mm = data.mm
    dd = data.dd
    hh = data.hh
    """ Función para la lectura de la imagen infrarroja llamandola desde los documentos"""
    NAME = str(yy[i]).zfill(2)+str(mm[i]).zfill(2)+str(dd[i]).zfill(2)+'-S'+str(hh[i]).zfill(2)
    hora = str(hh[i]).zfill(2)
    time = str(60*hh[i]).zfill(4)

    filename= 'D:/GPM_IMERG/3B-HHR.MS.MRG.3IMERG.'+NAME+'0000-E'+hora+'2959.'+time+'.V06B.HDF5.nc4'
    ds = nc.Dataset(filename)
    lat1 = ds.variables['lat'][:]
    lon1 = ds.variables['lon'][:]
    time1 = ds.variables['time'][:]
    Tb = ds.variables['precipitationCal'][0,:,:]
    imagen = np.array(Tb)
    pic1= imagen[:,:]
    pcp = np.transpose(pic1)
    return pcp
def multipoint(x, points2):
    y = points2.index[(points2['geometry']) == x][0]
    prueba = type(shape(points2['geometry'].iloc[y])) is Polygon
    if prueba is True:
        pnts = MultiPoint(list(x.exterior.coords))
    else:
        pnts = MultiPoint(list(shape(x).coords))
    return pnts
def pcp_radii(data,UMBRAL):
    data_base = gpd.GeoDataFrame(columns = ['geometry', 'ID'], crs = 'epsg:4326')
    nw, ne = np.zeros(len(data)), np.zeros(len(data))
    sw, se = np.zeros(len(data)), np.zeros(len(data))
    for ii in range(len(data)):
        result = process_imagen(data,ii, UMBRAL)
        data_base = pd.concat([data_base, result], axis=0)
        nw[ii], ne[ii], sw[ii], se[ii] = calculo_radios(ii,data,data_base)
    DF = pd.DataFrame()
    DF['ID'] = data.TC
    DF['dd'] = data.dd
    DF['mm'] = data.mm
    DF['yy'] = data.yy
    DF['hh'] = data.hh
    DF['lat'] = data.lat
    DF['lon']  = data.lon
    DF['MWS'] = data.MWS
    DF['RNE'] = ne
    DF['RNW'] = nw
    DF['RSW'] = sw
    DF['RSE'] = se
    DF = radio_prom(DF)
    return DF
def process_imagen(data,i,umbral):
    """ Función para el procesamiento final de las imagenes, con información de la
    data y un valor iterativo 'i'. Las respuestas finales se guardan en GeoDataBase."""
    imapru = correc(lec_imag(data,i))
    seg40 = bina(imapru,umbral)
    ctpos = [data['lon'][i],data['lat'][i]]
    r1 = data['RNE'][i]/111.1
    r2 = data['RNO'][i]/111.1
    r3 = data['RSO'][i]/111.1
    r4 = data['RSE'][i]/111.1
    arc = arco(r1,r2,r3,r4,ctpos)
    polyar = Polygon(list(zip(arc[0],arc[1])))
    con40 = contornos(seg40)
    selcon = filtcloud(con40,polyar)
    if selcon.empty:
        selcon['geometry'] = None
        selcon['ID']= (np.ones(selcon.shape[0])*i).astype(int)
        return selcon
    else:
        result = rectificacion(selcon,ctpos,data['Rp'][i]/111.1)
        result['ID']= (np.ones(result.shape[0])*i).astype(int)
        result = gdf_explode(result)
        indx = result['geometry'].apply(lambda nube: nube.intersects(polyar))
        return result[indx]
def radio_prom(df):
    RNE,RNW,RSE,RSW = list(df.RNE),list(df.RNW),list(df.RSE),list(df.RSW)
    r = np.zeros(np.size(RNE))
    for i in range(len(RNE)):
        pdf = pd.DataFrame(np.array([[RNE[i]],[RNW[i]],[RSE[i]],[RSW[i]]]))
        r[i]=pdf.loc[(pdf[0] > 0)].mean()[0]
    df['Rp']=r
    return df
def rectificacion(selcon,ctpos,r):
    """ Función que aplica un filtro en forma de castillo para poder quitar interacciones
    de los cumulos con la ZCIT (considerada por debajo de los 15°N). Los parametros de
    entrada representan los siguientes datos:
    selcon = contornos seleccionados
    ctpos = posicion del centro de la tormenta tomada de HURDAT
    r = es el radio ROUT en grados
    El resultado de la función representa la selección filtrada si el area del ROUT está
    interactuando con el area designada del ZCIT o deja los contornos intactos si no existe
    interacción teorica con el ZCIT"""
    ar = circulo(ctpos,r)
    polyar = Polygon(list(zip(ar[0],ar[1])))
    rec_inf = Polygon([(-130,0),(-130,15),(-10,15),(-10,0)]) #itcz
    a = rec_inf.intersects(polyar)
    pol_cast = Polygon([(-130,0),(-130,15),(ctpos[0]-r,15),(ctpos[0]-r,ctpos[1]-r),
                        (ctpos[0]+r,ctpos[1]-r),(ctpos[0]+r,15),(-10,15),(-10,0)])
    pol_cast = gpd.GeoDataFrame(index=[0], crs='EPSG:4326', geometry=[pol_cast])
    if a == True:
        filtro2 = gpd.overlay(selcon, pol_cast, how='difference')
        return filtro2
    else:
        if ctpos[1] > 15:
            pol_cast = Polygon([(-130,0),(-130,15.1),(ctpos[0]-r,15.1),(ctpos[0]-r,ctpos[1]-r),
                        (ctpos[0]+r,ctpos[1]-r),(ctpos[0]+r,15.1),(-10,15.1),(-10,0)])
            pol_cast = gpd.GeoDataFrame(index=[0], crs='EPSG:4326', geometry=[pol_cast])
            filtro2 = gpd.overlay(selcon, pol_cast, how='difference')
            return filtro2
        else:
            return selcon
def save_df(direccion,datos,nw,ne,sw,se):
    df = pd.read_csv(direccion+'/'+datos, sep='\n', skip_blank_lines=True, header = None)
    columns = (df[0].str.split(r',',expand= True))
    columns[12],columns[13],columns[14],columns[15] = nw, ne, sw, se
    columns = trim_all_columns(columns)
    columns[2] = cambiar_sufijos(columns,2)
    columns[3] = cambiar_sufijos(columns,3)
    columns[12] = columns[12].apply(data_text)
    columns[13] = columns[13].apply(data_text)
    columns[14] = columns[14].apply(data_text)
    columns[15] = columns[15].apply(data_text)
    c = [0,1,2,3,4,5,7,11,12,13,14,15]
    columns = columns[c]
    crear_carp(direccion+'extend')
    columns.to_csv(direccion+'extend/'+datos,sep = '\t', columns = c ,index = False, header = False)
def trim_all_columns(df):
    """
    Trim whitespace from ends of each value across all series in dataframe
    """
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x
    return df.applymap(trim_strings)
