import os
import numpy as np
import pydicom
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
from datetime import datetime

class GestorDICOM:
    def __init__(self, ruta_carpeta):
        """
        Inicializa la clase de GestorDICOM con la ruta de la carpeta que queramos usar
        Crea los atributos principales para almacenar imagenes, crear el volumen 3D,
        espaciado y espesor
        """
        self.ruta_carpeta = ruta_carpeta
        self.lista_slices = []
        self.volumen = None
        self.espaciado_pixel = None
        self.espesor_corte = None

    def _obtener_archivos_dcm(self):
        """
        Busca archivos DICOM en la carpeta
        """
        archivos = []
        for raiz, _, nombres in os.walk(self.ruta_carpeta):
            for n in nombres:
                if n.lower().endswith('.dcm'):
                    archivos.append(os.path.join(raiz, n))
        return archivos

    def cargar_imagenes(self):
        """
        Carga y ordena los archivos DICOM
        """
        archivos = self._obtener_archivos_dcm()
        if len(archivos) == 0:
            raise FileNotFoundError("No se encontraron archivos DICOM.")

        datasets = []
        for a in archivos:
            try:
                ds = pydicom.dcmread(a)
                datasets.append(ds)
            except:
                continue

    def cargar_imagenes(self):
        """
        Carga y ordena los archivos DICOM
        """
        archivos = self._obtener_archivos_dcm()
        if len(archivos) == 0:
            raise FileNotFoundError("No se encontraron archivos DICOM.")

        datasets = []
        for a in archivos:
            try:
                ds = pydicom.dcmread(a)
                datasets.append(ds)
            except:
                continue

    def cargar_imagenes(self):
        """
        Carga y ordena los archivos DICOM
        """
        archivos = self._obtener_archivos_dcm()
        if len(archivos) == 0:
            raise FileNotFoundError("No se encontraron archivos DICOM.")

        datasets = []
        for a in archivos:
            try:
                ds = pydicom.dcmread(a)
                datasets.append(ds)
            except:
                continue

        def clave_orden(ds):
            """
            Ordena los cortes DICOM segun su posicion o numero de instancia
            y obtiene el espaciado de pixel y el espesor del corte
            """
            if hasattr(ds, "InstanceNumber"):
                return int(ds.InstanceNumber)
            elif hasattr(ds, "ImagePositionPatient"):
                return float(ds.ImagePositionPatient[2])
            elif hasattr(ds, "SliceLocation"):
                return float(ds.SliceLocation)
            else:
                return 0

        datasets.sort(key=clave_orden)
        self.lista_slices = datasets

        primero = datasets[0]
        self.espaciado_pixel = [float(primero.PixelSpacing[0]), float(primero.PixelSpacing[1])] if hasattr(primero, "PixelSpacing") else [1, 1]
        self.espesor_corte = float(primero.SliceThickness) if hasattr(primero, "SliceThickness") else 1

    def construir_volumen(self):
        """
        Construye la matriz 3D. Convierte cada imagen 2D en
        un arreglo de NumPy de tipo float32 y las apila en el eje z
        """
        if len(self.lista_slices) == 0:
            self.cargar_imagenes()

        imagenes = []
        for ds in self.lista_slices:
            try:
                arr = ds.pixel_array.astype(np.float32)
                imagenes.append(arr)
            except:
                pass

        self.volumen = np.stack(imagenes, axis=0)
        return self.volumen

    def obtener_dataframe_datos(self):
        """
        Crea un DataFrame con los data elements DICOM
        """
        registros = []
        for ds in self.lista_slices:
            datos = {}
            for elem in ds:
                if elem.VR != 'SQ':
                    datos[str(elem.keyword)] = str(elem.value)
            registros.append(datos)
        df = pd.DataFrame(registros)
        return df
    class EstudioImaginologico:
    def __init__(self, gestor_dicom, nombre_estudio="Estudio"):
        """
        Inicializa la clase para el objeto de estudio imaginologico
        y construye o asigna el volumen 3D y guarda su forma
        """
        self.nombre_estudio = nombre_estudio
        self.gestor = gestor_dicom
        self.volumen = gestor_dicom.volumen if gestor_dicom.volumen is not None else gestor_dicom.construir_volumen()

        primer = gestor_dicom.lista_slices[0]
        
        self.StudyDate = getattr(primer, "StudyDate", "")
        self.StudyTime = getattr(primer, "StudyTime", "")
        self.Modality = getattr(primer, "Modality", "")
        self.StudyDescription = getattr(primer, "StudyDescription", "")
        self.SeriesTime = getattr(primer, "SeriesTime", "")
        self.DurationSeconds = self._calcular_duracion(self.StudyTime, self.SeriesTime)
        self.ImageShape = self.volumen.shape

    def _calcular_duracion(self, hora_ini, hora_fin):
        """
        Calcula la duracion en segundos a partir de las horas dadas
        """
        try:
            if not hora_ini or not hora_fin:
                return None
            t1 = datetime.strptime(hora_ini.split('.')[0], "%H%M%S")
            t2 = datetime.strptime(hora_fin.split('.')[0], "%H%M%S")
            return abs((t2 - t1).total_seconds())
        except:
            return None
