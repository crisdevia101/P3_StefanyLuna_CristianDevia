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
            
    def _normalizar_a_uint8(self, imagen):
        minimo, maximo = np.min(imagen), np.max(imagen)
        if maximo == minimo:
            return np.zeros_like(imagen, dtype=np.uint8)
        normalizada = ((imagen - minimo) / (maximo - minimo)) * 255
        return normalizada.astype(np.uint8)

    def _asegurar_carpeta(self, nombre_subcarpeta):
        """Crea carpetas dentro de resultados/ según la función."""
        ruta_carpeta = os.path.join("resultados", nombre_subcarpeta)
        os.makedirs(ruta_carpeta, exist_ok=True)
        return ruta_carpeta

    def mostrar_cortes_ortogonales(self, indice=None):
        """
        Muestra y guarda los tres cortes ortogonales (transversal, coronal y sagital)
        del volumen 3D en una sola figura
        """
        z, h, w = self.volumen.shape
        if indice is None:
            indice = z // 2

        corte_transversal = self.volumen[indice, :, :]
        corte_coronal = self.volumen[:, :, w // 2]
        corte_sagital = self.volumen[:, h // 2, :]

        fig, ejes = plt.subplots(1, 3, figsize=(10, 4))
        ejes[0].imshow(corte_transversal, cmap='gray')
        ejes[0].set_title("Transversal")
        ejes[1].imshow(corte_coronal, cmap='gray')
        ejes[1].set_title("Coronal")
        ejes[2].imshow(corte_sagital, cmap='gray')
        ejes[2].set_title("Sagital")
        for ax in ejes:
            ax.axis("off")
        plt.suptitle(self.nombre_estudio)

        ruta_carpeta = self._asegurar_carpeta("cortes")
        nombre_archivo = os.path.join(ruta_carpeta, f"cortes_{self.nombre_estudio}.png")
        plt.savefig(nombre_archivo, bbox_inches='tight')
        plt.show()
        print("Cortes ortogonales guardados en:", nombre_archivo)

    def zoom_y_guardar(self, indice_z, x1, y1, x2, y2, nombre_salida="recorte.png"):
        """
        Realiza un recorte (zoom) del corte seleccionado, muestra sus dimensiones
        en milimetros y guarda el resiltado en la carpeta "zoom"
        """
        corte = self.volumen[indice_z, :, :]
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        h, w = corte.shape
        x_min = max(0, min(x_min, w - 1))
        x_max = max(0, min(x_max, w))
        y_min = max(0, min(y_min, h - 1))
        y_max = max(0, min(y_max, h))

        if x_max <= x_min or y_max <= y_min:
            print("Coordenadas de recorte inválidas.")
            return

        recorte = corte[y_min:y_max, x_min:x_max]
        recorte_norm = self._normalizar_a_uint8(recorte)

        alto_pix, ancho_pix = recorte_norm.shape
        esp_h, esp_w = self.gestor.espaciado_pixel
        esp_z = self.gestor.espesor_corte

        ancho_mm = ancho_pix * esp_w
        alto_mm = alto_pix * esp_h
        espesor_mm = esp_z

        recorte_bgr = cv2.cvtColor(recorte_norm, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(recorte_bgr, (0, 0), (ancho_pix - 1, alto_pix - 1), (0, 255, 0), 2)
        texto = f"W={ancho_mm:.1f}mm H={alto_mm:.1f}mm T={espesor_mm:.1f}mm"
        cv2.putText(recorte_bgr, texto, (5, alto_pix - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        recorte_redimensionado = cv2.resize(recorte_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)

        ruta_carpeta = self._asegurar_carpeta("zoom")
        ruta_guardado = os.path.join(ruta_carpeta, nombre_salida)
        cv2.imwrite(ruta_guardado, recorte_redimensionado)
        print("Recorte guardado en:", ruta_guardado)

        fig, ejes = plt.subplots(1, 2, figsize=(8, 4))
        ejes[0].imshow(corte, cmap='gray')
        ejes[0].set_title("Corte original")
        ejes[0].axis("off")
        ejes[1].imshow(cv2.cvtColor(recorte_redimensionado, cv2.COLOR_BGR2RGB))
        ejes[1].set_title("Recorte guardado")
        ejes[1].axis("off")
        plt.show()
