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
