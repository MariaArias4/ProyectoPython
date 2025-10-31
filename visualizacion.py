import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ProyectoPy.utilidades import preparar_conteos_semanales

# Tema visual de seaborn
sns.set_theme(style="whitegrid")

# Función para mostrar tres gráficos a partir del DataFrame de incidencias
def plot_resumen(df):
    # Si no hay datos, mostrar mensaje y salir
    if df.empty:
        plt.figure()
        plt.text(0.5, 0.5, "No hay datos para graficar", ha='center', va='center')
        plt.axis('off')
        plt.show()
        return

    # ---------------------------
    # Incidencias totales por semana
    # ---------------------------
    # Preparar conteos semanales por ubicación usando la utilidad
    conteos = preparar_conteos_semanales(df)

    # Si hay conteos válidos, agrupar por year-semana y sumar para obtener el total por semana
    if not conteos.empty:
        per_week = conteos.groupby(['year', 'semana'])['n'].sum().reset_index()
        per_week = per_week.sort_values(['year', 'semana'])
        # Columna x para etiquetar el eje X como "YYYY-S" (por ejemplo "2023-12")
        per_week['x'] = per_week['year'].astype(str) + '-' + per_week['semana'].astype(str)

        plt.figure(figsize=(10, 4))
        sns.lineplot(data=per_week, x='x', y='n', marker='o')
        plt.xticks(rotation=45)
        plt.xlabel('Semana')
        plt.ylabel('Número de incidencias')
        plt.title('Incidencias totales por semana')
        plt.tight_layout()
        plt.show()
    else:
        # Mensaje cuando no hay fechas válidas en los datos originales
        plt.figure()
        plt.text(0.5, 0.5, "No hay fechas válidas para conteo semanal", ha='center', va='center')
        plt.axis('off')
        plt.show()

    # ---------------------------
    # Top ubicaciones por número de incidencias
    # ---------------------------
    # Contar incidencias por ubicación y tomar las 10 principales
    top_ubicaciones = df['Ubicación'].value_counts().nlargest(10)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=top_ubicaciones.values, y=top_ubicaciones.index, palette="Blues_d")
    plt.xlabel('Número de incidencias')
    plt.ylabel('Ubicación')
    plt.title('Top 10 ubicaciones por número de incidencias')
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # Mapa de calor: Ubicación vs Tipo
    # ---------------------------
    # Crear tabla cruzada (pivot) con conteos por ubicación y tipo
    tabla_pivot = pd.crosstab(df['Ubicación'], df['Tipo'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(tabla_pivot, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Tipo')
    plt.ylabel('Ubicación')
    plt.title('Mapa de calor: Ubicación vs Tipo (conteo)')
    plt.tight_layout()
    plt.show()
