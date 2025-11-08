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

