import pandas as pd

COLUMNAS = ["Nombre", "Descripción", "Fecha", "Ubicación", "Tipo"]

# Convierte una lista de diccionarios (cada diccionario es una fila) a un DataFrame
def tabla_lista_a_dataframe(lista_dicts):
    # Crear DataFrame
    df = pd.DataFrame(lista_dicts, columns=COLUMNAS)
    # Convertir la columna 'Fecha' a datetime; errores se convierten a NaT
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    # Asegurar que el resto de columnas son texto
    df['Nombre'] = df['Nombre'].astype(str)
    df['Descripción'] = df['Descripción'].astype(str)
    df['Ubicación'] = df['Ubicación'].astype(str)
    df['Tipo'] = df['Tipo'].astype(str)
    return df

# A partir de un DataFrame con 'Fecha' (datetime) y 'Ubicación',
# devuelve un DataFrame con columnas: year, semana, ubicacion, n (conteo)
def preparar_conteos_semanales(df):
    # Eliminar filas sin fecha válida para evitar NaT en los cálculos
    df2 = df.dropna(subset=['Fecha']).copy()

    # Extraer año a partir de la fecha
    df2['year'] = df2['Fecha'].dt.year

    # Extraer número de semana ISO a partir de la fecha
    df2['semana'] = df2['Fecha'].dt.isocalendar().week

    # Agrupar por año, semana y ubicación y contar registros por grupo
    conteos = df2.groupby(['year', 'semana', 'Ubicación']).size().reset_index(name='n')

    # Renombrar la columna 'Ubicación' a 'ubicacion' para consistencia en minúsculas
    conteos = conteos.rename(columns={'Ubicación': 'ubicacion'})

    # Ordenar por ubicación, año y semana para facilitar procesamientos posteriores
    conteos = conteos.sort_values(['ubicacion', 'year', 'semana']).reset_index(drop=True)

    # Devolver el DataFrame de conteos semanales por ubicación
    return conteos
