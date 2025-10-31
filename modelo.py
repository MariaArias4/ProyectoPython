# modelo.py
import math
from collections import defaultdict

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error

from ProyectoPy.utilidades import preparar_conteos_semanales

# Función principal de predicción
def predecir(df, semanas_a_predecir=4, usar_clustering=True, n_clusters=3, modelo_alternativo='rf'):
    # informe acumulado en líneas
    informe_lines = []
    predicciones = {}

    # Comprobar si hay datos
    if df is None or df.empty:
        return "No hay datos disponibles.", {}

    # 1) Resumen por Tipo
    tipos = df['Tipo'].fillna("Desconocido").astype(str)
    conteo_tipos = tipos.value_counts()
    total = len(tipos)
    informe_lines.append("Distribución por Tipo:")
    for t, c in conteo_tipos.items():
        informe_lines.append(f" - {t}: {c} ({c/total*100:.1f}%)")

    # 2) Preparar conteos semanales
    conteos = preparar_conteos_semanales(df)  # year, semana, ubicacion, n
    if conteos.empty:
        informe_lines.append("No hay datos semanales suficientes para modelado.")
        return "\n".join(informe_lines), {}

    # 3) Crear features adicionales por ubicación
    conteos = conteos.sort_values(['ubicacion', 'year', 'semana']).reset_index(drop=True)

    # feature: prev_n (semana anterior)
    conteos['prev_n'] = conteos.groupby('ubicacion')['n'].shift(1)

    # feature: media móvil corta (ma3) — media de las últimas 3 semanas por ubicación
    conteos['ma3'] = (
        conteos.groupby('ubicacion')['n']
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # feature: media histórica por ubicación
    conteos['media_ubic'] = conteos.groupby('ubicacion')['n'].transform('mean')

    # feature: semana del año (numérica, permite capturar sazonalidad simple)
    conteos['weekofyear'] = conteos['semana'].astype(int)

    # filas utilizables para entrenamiento (prev_n no nulo)
    datos_modelo = conteos.dropna(subset=['prev_n']).copy()
    if datos_modelo.empty:
        informe_lines.append("No hay suficientes pares (semana anterior, actual) por ubicación para entrenar.")
        preds = _prediccion_media_movil(conteos, semanas_a_predecir)
        informe_lines.append("Predicción: se aplicó media móvil por falta de datos para regresión.")
        return "\n".join(informe_lines), preds

    # columnas de features a usar
    feat_cols = ['prev_n', 'ma3', 'media_ubic', 'weekofyear']
    X = datos_modelo[feat_cols].values
    y = datos_modelo['n'].values

    # 4) Split temporal aproximado (últimas 4 semanas como test)
    datos_modelo['week_index'] = datos_modelo['year'].astype(int) * 100 + datos_modelo['semana'].astype(int)
    unique_weeks = sorted(datos_modelo['week_index'].unique())
    if len(unique_weeks) > 4:
        split_week = unique_weeks[-4]
        train_mask = datos_modelo['week_index'] < split_week
        test_mask = datos_modelo['week_index'] >= split_week
        # asegurar mínimo razonable de train
        if train_mask.sum() < 3:
            train_mask = pd.Series([True] * len(datos_modelo))
            test_mask = ~train_mask
    else:
        # si pocas semanas, no separamos (todo entrenamiento)
        train_mask = pd.Series([True] * len(datos_modelo))
        test_mask = pd.Series([False] * len(datos_modelo))

    X_train = X[train_mask.values]
    y_train = y[train_mask.values]
    X_test = X[test_mask.values] if test_mask.any() else None
    y_test = y[test_mask.values] if test_mask.any() else None

    # si muy pocos puntos de entrenamiento, fallback a media móvil
    if len(y_train) < 3:
        preds = _prediccion_media_movil(conteos, semanas_a_predecir)
        informe_lines.append("Datos de entrenamiento insuficientes para modelos; aplicado fallback de media móvil.")
        return "\n".join(informe_lines), preds
    
    # 5) Entrenar modelo (lineal o alternativo)
    modelo = None
    if modelo_alternativo == 'rf':
        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Evaluación en train
    y_train_pred = modelo.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred) if len(y_train) > 1 else float('nan')
    mae_train = mean_absolute_error(y_train, y_train_pred) if len(y_train) > 0 else float('nan')

    informe_lines.append("")
    informe_lines.append(f"Modelo entrenado ({'RandomForest' if modelo_alternativo=='rf' else 'LinearRegression'}) — features: {', '.join(feat_cols)}")
    informe_lines.append(f"R2 (train): {r2_train:.3f}    MAE (train): {mae_train:.2f}")

    # Evaluación en test temporal si existe
    if X_test is not None and len(X_test) > 0:
        y_test_pred = modelo.predict(X_test)
        r2_test = r2_score(y_test, y_test_pred) if len(y_test) > 1 else float('nan')
        mae_test = mean_absolute_error(y_test, y_test_pred) if len(y_test) > 0 else float('nan')
        informe_lines.append(f"R2 (test temporal, ~últimas semanas): {r2_test:.3f}    MAE (test): {mae_test:.2f}")

    # 6) Generar predicciones iterativas por ubicación
    # Usamos floats y no redondeamos durante las iteraciones; solo al mostrar podrías formatear
    ultimos = conteos.groupby('ubicacion').tail(1).set_index('ubicacion')
    for ubicacion, fila in ultimos.iterrows():
        # preparar vector inicial de features con el último registro conocido
        prev = float(fila['n'])  # prev_n inicial
        ma3 = float(fila.get('ma3', prev))
        media_ubic = float(fila.get('media_ubic', prev))
        week = int(fila.get('semana', 0))

        arr = []
        # predecir semanas_a_predecir iterativamente, actualizando features dependientes
        for step in range(semanas_a_predecir):
            # construir vector de features para predicción
            feat_vector = [prev, ma3, media_ubic, ((week + step - 1) % 52) + 1]  # weekofyear aproximado
            p = modelo.predict([feat_vector])[0]
            p = max(0.0, float(p))  # evitar negativos, mantener float

            arr.append(p)

            # actualizar prev y ma3 para la próxima iteración (media móvil simple)
            prev = p
            # actualizar ma3: aproximación simple: desplazar ventana con peso igual
            ma3 = (ma3 * 2 + p) / 3.0

        predicciones[ubicacion] = arr

    # 7) Resumen top ubicaciones
    top = conteos.groupby('ubicacion')['n'].sum().sort_values(ascending=False).head(6)
    informe_lines.append("")
    informe_lines.append("Top ubicaciones (total incidencias en periodo):")
    for u, s in top.items():
        informe_lines.append(f" - {u}: {int(s)} incidencias")

    # 8) Clustering por ubicación (opcional)
    if usar_clustering:
        cluster_summary = _clusters_por_ubicacion(conteos, n_clusters=n_clusters)
        informe_lines.append("")
        informe_lines.append(f"Clustering (KMeans, k={n_clusters}) — resumen por cluster:")
        for k, info in cluster_summary.items():
            informe_lines.append(f" - Cluster {k}: {info['n_ubicaciones']} ubicaciones; media semanal aproximada {info['media_n']:.1f}")

    # 9) Mostrar primeras predicciones (convertir a int al mostrar es opcional)
    informe_lines.append("")
    informe_lines.append(f"Predicción (próximas {semanas_a_predecir} semanas) — primeras ubicaciones:")
    i = 0
    for u, arr in predicciones.items():
        # mostrar arr con un decimal o como entero redondeado según preferencia
        muestra = [round(v, 2) for v in arr]
        informe_lines.append(f" - {u}: {muestra}")
        i += 1
        if i >= 6:
            break

    return "\n".join(informe_lines), predicciones


# Predicción por media móvil (fallback)
def _prediccion_media_movil(conteos_sem, semanas=4):
    preds = {}
    if conteos_sem.empty:
        return preds
    conteos_sem = conteos_sem.copy()
    conteos_sem['week_index'] = conteos_sem['year'].astype(int) * 100 + conteos_sem['semana'].astype(int)
    for u in conteos_sem['ubicacion'].unique():
        s = conteos_sem[conteos_sem['ubicacion'] == u].sort_values('week_index')
        if len(s) >= 3:
            base = float(s['n'].tail(3).mean())
        elif len(s) >= 1:
            base = float(s['n'].iloc[-1])
        else:
            base = 0.0
        preds[u] = [max(0.0, base) for _ in range(semanas)]
    return preds


# Clustering por ubicación (resumen)
def _clusters_por_ubicacion(conteos_sem, n_clusters=3):
    resumen = conteos_sem.groupby('ubicacion')['n'].agg(['sum', 'mean', 'std']).fillna(0.0)
    if len(resumen) <= 1:
        return {0: {'n_ubicaciones': len(resumen),
                    'media_n': float(resumen['mean'].mean() if not resumen.empty else 0.0)}}

    X = resumen[['sum', 'mean', 'std']].values
    k = min(n_clusters, max(1, len(resumen) - 1))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    resumen['cluster'] = kmeans.labels_
    clusters = {}
    for cluster_id, grupo in resumen.groupby('cluster'):
        clusters[cluster_id] = {
            'n_ubicaciones': int(len(grupo)),
            'media_n': float(grupo['mean'].mean())
        }
    return clusters
