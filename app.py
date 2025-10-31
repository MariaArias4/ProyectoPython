# app.py
import sys
import os
import csv
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QTableWidget, QTableWidgetItem,
    QLabel, QMessageBox, QTextEdit, QVBoxLayout, QAbstractItemView, QHBoxLayout)
from PyQt5.QtGui import QFont #type: ignore
from PyQt5.QtCore import Qt

from ProyectoPy.utilidades import tabla_lista_a_dataframe
from ProyectoPy.visualizacion import plot_resumen
from ProyectoPy.modelo import predecir

# Constantes
RUTA_CSV_POR_DEFECTO = "dataset.csv"
COLUMNAS = ["Nombre", "Descripción", "Fecha", "Ubicación", "Tipo"]

# Función para cargar CSV en la tabla
def rellenar_tabla(ruta, tabla):
    # Limpiar tabla antes de cargar nuevos datos
    tabla.setRowCount(0)
    # Leer CSV y llenar la tabla
    with open(ruta, newline='', encoding='utf-8') as f:
        lector = csv.DictReader(f)
        # Rellenar la tabla fila por fila
        for fila in lector:
            numFila = tabla.rowCount()
            tabla.insertRow(numFila)
            valores = [
                fila.get("Nombre", ""),
                fila.get("Descripción", ""),
                fila.get("Fecha", ""),
                fila.get("Ubicación", ""),
                fila.get("Tipo", "")
            ]
            # Rellenar cada celda de la fila actual con su texto y centrar la columna Fecha 
            for c, texto in enumerate(valores):
                item = QTableWidgetItem(texto)
                if c == 2:
                    item.setTextAlignment(Qt.AlignCenter)
                # marcar la celda como no editable
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                tabla.setItem(numFila, c, item)

#Creacion de la ventana principal
class VentanaPrincipal(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Incidencias")
        self.setGeometry(500,200,800,300)

        # Layout principal
        self.layout_principal = QVBoxLayout(self)
        self.setLayout(self.layout_principal)

        # Título
        self.labelTitulo = QLabel("Predicción de incidencias", self)
        font = QFont("Arial", 23, QFont.Bold)
        font.setUnderline(True)
        self.labelTitulo.setFont(font)
        self.labelTitulo.setAlignment(Qt.AlignCenter)
        self.layout_principal.addWidget(self.labelTitulo)
        
        # Tabla no editable
        self.tabla = QTableWidget(self)
        self.tabla.horizontalHeader().setDefaultSectionSize(150)
        self.tabla.setColumnCount(len(COLUMNAS))
        self.tabla.setHorizontalHeaderLabels(COLUMNAS)
        self.tabla.setAlternatingRowColors(True)
        self.tabla.setMinimumHeight(300)
        self.tabla.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.layout_principal.addWidget(self.tabla)

       # layout horizontal para los botones
        hbox_botones = QHBoxLayout()
        # botón izquierdo (Gráficos)
        self.boton_graficos = QPushButton("Gráficos", self)
        self.boton_graficos.setFixedSize(120, 30)
        self.boton_graficos.setFont(QFont("Arial", 12, QFont.Bold))
        hbox_botones.addWidget(self.boton_graficos)
        
        hbox_botones.addStretch(1)

        # botón derecho (Informe) alineado a la derecha
        self.boton_informe = QPushButton("Informe de predicción", self)
        self.boton_informe.setFixedSize(180, 30)
        self.boton_informe.setFont(QFont("Arial", 12, QFont.Bold))
        hbox_botones.addWidget(self.boton_informe)

        self.layout_principal.addLayout(hbox_botones)

        # Caja para informe
        self.caja_informe = QTextEdit(self)
        self.caja_informe.setReadOnly(True)
        self.caja_informe.setPlaceholderText("Aquí aparecerá el informe de predicción.")
        self.caja_informe.setMinimumHeight(120)
        self.layout_principal.addWidget(self.caja_informe)

        # Conexiones
        self.boton_graficos.clicked.connect(self.mostrar_graficos)
        self.boton_informe.clicked.connect(self.generar_informe)

        # Cargar CSV por defecto al iniciar
        self._cargar_csv_por_defecto_al_iniciar()

    def _cargar_csv_por_defecto_al_iniciar(self):
        if os.path.exists(RUTA_CSV_POR_DEFECTO):
            try:
                rellenar_tabla(RUTA_CSV_POR_DEFECTO, self.tabla)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al leer {RUTA_CSV_POR_DEFECTO}:\n{e}")
        else:
            QMessageBox.warning(self, "CSV no encontrado", f"No se encontró '{RUTA_CSV_POR_DEFECTO}'.\nColoca tu CSV con ese nombre en la carpeta.")
            
    def mostrar_graficos(self):
        lista = self.obtener_datos_como_lista_dicts()
        df = tabla_lista_a_dataframe(lista)
        plot_resumen(df)

    def generar_informe(self):
        lista = self.obtener_datos_como_lista_dicts()
        df = tabla_lista_a_dataframe(lista)
        informe_texto, predicciones = predecir(df)
        self.caja_informe.setPlainText(informe_texto)
        QMessageBox.information(self, "Informe de predicción", "Informe generado y mostrado en la ventana.")

    def obtener_datos_como_lista_dicts(self):
        datos = []
        for r in range(self.tabla.rowCount()):
            fila = {}
            for c, col in enumerate(COLUMNAS):
                item = self.tabla.item(r, c)
                val = item.text().strip() if item else ""
                if col == "Fecha" and val:
                    try:
                        d = datetime.fromisoformat(val)
                        val = d.strftime("%Y-%m-%d")
                    except Exception:
                        pass
                fila[col] = val
            datos.append(fila)
        return datos

def main():
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
