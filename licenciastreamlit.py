

import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

"""# **Unión de Trimestres**"""

import os
import pandas as pd
import gc

# 1. DEFINICIÓN DE ESTRUCTURA Y DICCIONARIOS
columnas_licencias = [
    "CodigoIsapre", "FechaInformacion", "RunTrabajador", "FechaEmisionLicencia", "NumeroDias",
    "FechaInicioReposo", "EdadTrabajador", "SexoTrabajador", "ActividadLaboral", "Ocupacion",
    "TipoLicencia", "CaracteristicasReposo", "RunProfesional", "TipoProfesional",
    "TipoLicenciaContraloria", "NumeroDiasAutorizados", "DiagnosticoPrincipal", "TipoResolucion",
    "Periodo", "ReposoAutorizado", "DerechoSubsidio", "FechaRecepcionIsapre", "FechaResolucionIsapre",
    "FechaRecepcionEmpleador", "Region", "CalidadTrabajador", "EntidadPagadora", "NumeroDiasPagar",
    "MontoSubsidioLiquido", "MontoAportePrevisionalIsapre", "FechaInicioPago", "Recuperabilidad",
    "FechaConcepcion", "MontoAportePrevisionalPensiones", "OtrosDiagnosticos", "RunHijo",
    "LugarReposo", "CausaRechazoModificacion", "NumeroDiasPreviosAutorizados",
    "FechaPrimeraAfiliacionPrevisional", "FechaContratoTrabajo", "MontoBaseCalculoSubsidio",
    "RutEmpleador", "FechaNacimientoHijo", "eliminar"
]

# Creación de categorias
enfermedades = {'A00-B99': 'Enfermedades infecciosas y parasitarias', 'C00-D48': 'Tumores (neoplasias)', 'D50-D89': 'Enfermedades de la sangre y órganos hematopoyéticos', 'E00-E90': 'Enfermedades endocrinas, nutricionales y metabólicas', 'F00-F99': 'Trastornos mentales y del comportamiento', 'G00-G99': 'Enfermedades del sistema nervioso', 'H00-H59': 'Enfermedades del ojo y anexos', 'H60-H95': 'Enfermedades del oído y mastoides', 'I00-I99': 'Enfermedades del sistema circulatorio', 'J00-J99': 'Enfermedades del sistema respiratorio', 'K00-K93': 'Enfermedades del sistema digestivo', 'L00-L99': 'Enfermedades de la piel y tejido subcutáneo', 'M00-M99': 'Enfermedades osteomusculares y tejido conectivo', 'N00-N99': 'Enfermedades genitourinarias', 'O00-O99': 'Embarazo, parto y puerperio', 'P00-P96': 'Afecciones originadas en el período perinatal', 'Q00-Q99': 'Malformaciones congénitas', 'R00-R99': 'Síntomas, signos y hallazgos anormales', 'S00-T98': 'Traumatismos y envenenamientos', 'V01-Y98': 'Causas externas de morbilidad y mortalidad', 'Z00-Z99': 'Factores que influyen en el estado de salud'}
grupo_etario = {'0-5': 'Niñez', '16-20': 'Jóvenes', '21-25': 'Jóvenes', '26-30': 'Adultos Jóvenes', '31-35': 'Adultos Jóvenes', '36-40': 'Adultos', '41-45': 'Adultos', '46-50': 'Adultos', '51-55': 'Adultos', '56-60': 'Adultos', '61-65': 'Adultos Mayores', '66-70': 'Adultos Mayores', '71-75': 'Tercera Edad', '76-80': 'Tercera Edad', '81-85': 'Tercera Edad', '86-90': 'Tercera Edad', '91-95': 'Tercera Edad'}
diccionario_ley = {'Trab. Pub. Afecto 18.834': '1. Públicos Afectos', 'Trab. Pub. No Afecto 18.834': '2. Públicos No Afectos', 'Trab. Dep. Sector Privado': '3. El Resto (Privado/Indep)', 'Trab. Independiente': '3. El Resto (Privado/Indep)', 'Z. Sin Clasificar': '3. El Resto (Privado/Indep)'}
isapres = {99: "Banmédica", 67: "Colmena", 107: "Consalud", 78: "Cruz Blanca", 94: "Cruz del Norte", 108: "Esencial", 76: "Fundación", 63: "Isalud", 81: "Nueva Masvida", 80: "Vida Tres"}

# 2. FUNCIÓN DE PROCESAMIENTO AUTOMÁTICO
def procesar_y_limpiar(df):
    # Aplicar columnas nuevas
    df['CategoriaDiagnostico'] = df['DiagnosticoPrincipal'].map(enfermedades)
    df['GrupoEtario'] = df['EdadTrabajador'].map(grupo_etario)
    df['Clasificacion_Ley'] = df['CalidadTrabajador'].map(diccionario_ley)
    df['Isapres'] = df['CodigoIsapre'].map(isapres)

    # Rellenar nulos
    df['CategoriaDiagnostico'] = df['CategoriaDiagnostico'].fillna("Sin diagnostico")
    df['GrupoEtario'] = df['GrupoEtario'].fillna("Sin información")
    df['Clasificacion_Ley'] = df['Clasificacion_Ley'].fillna("Sin información")
    df['OtrosDiagnosticos'] = df['OtrosDiagnosticos'].fillna("Sin diagnostico")
    df['DiagnosticoPrincipal'] = df['DiagnosticoPrincipal'].fillna("Sin diagnostico")

    # Lista final de columnas
    columnas_finales = ['CodigoIsapre', 'FechaInformacion', 'RunTrabajador', 'NumeroDias',
                        'EdadTrabajador', 'SexoTrabajador', 'ActividadLaboral', 'TipoLicencia',
                        'CaracteristicasReposo', 'RunProfesional', 'TipoProfesional',
                        'TipoLicenciaContraloria', 'NumeroDiasAutorizados', 'DiagnosticoPrincipal',
                        'TipoResolucion', 'Periodo', 'Region', 'CalidadTrabajador',
                        'EntidadPagadora', 'NumeroDiasPagar', 'MontoSubsidioLiquido',
                        'OtrosDiagnosticos', 'CausaRechazoModificacion', 'FechaContratoTrabajo',
                        'MontoBaseCalculoSubsidio', 'RutEmpleador', 'CategoriaDiagnostico',
                        'GrupoEtario', 'Clasificacion_Ley','Isapres']

    # Filtrar columnas existentes
    return df[[c for c in columnas_finales if c in df.columns]]

# 3. CICLO DE CARGA
ruta_base = r"C:\Users\arena\OneDrive - Corporación Santo Tomas\7mo semestre\MACHINE LEARNING\PROYECTO\CARPETAS ZIP"
lista_dfs = []

for archivo in os.listdir(ruta_base):
    if archivo.endswith(".txt"):
        print(f"Procesando: {archivo}")
        df_temp = pd.read_csv(os.path.join(ruta_base, archivo), sep="|", header=None, names=columnas_licencias, encoding="latin-1", on_bad_lines="skip")

        # Aplicamos las clasificaciones
        lista_dfs.append(procesar_y_limpiar(df_temp))

        del df_temp
        gc.collect()

# 4. UNIÓN FINAL
df_licencias_optimizada = pd.concat(lista_dfs, ignore_index=True)
print(f"Proceso finalizado. Total filas: {df_licencias_optimizada.shape[0]}")
display(df_licencias_optimizada.head(5).T)

"""## Revisión de NULLS"""

# Visualización de nulls por columna
print("Nulos por columna:")
print(df_licencias_optimizada.isnull().sum())

# Cuántos nulos hay por columna
nulos = df_licencias_optimizada.isnull().sum()
print(nulos[nulos > 0])

# Cuántas filas totales tiene tu base final
print(f"Total de filas en la unión: {df_licencias_optimizada.shape[0]}")

# Verificación de que estén todos los trimestres por año
print(df_licencias_optimizada['FechaInformacion'].unique())

"""## Revisión de datos duplicados"""

# Verificar los duplicados al unir los trimestres
duplicados = df_licencias_optimizada.duplicated().sum()
print(f"Se encontraron {duplicados} filas duplicadas.")

# Ver una muestra de 10 registros duplicados
duplicados = df_licencias_optimizada[df_licencias_optimizada.duplicated(keep=False)]
display(duplicados.head(10))

# Eliminar los duplicados
df_licencias_optimizada.drop_duplicates(inplace=True)

# Verificar cuántas filas quedaron
print(f"Ahora tu base tiene {df_licencias_optimizada.shape[0]} filas.")

# Comprobación de que los duplicados fueron eliminados
duplicados = df_licencias_optimizada.duplicated().sum()
print(f"Se encontraron {duplicados} filas duplicadas.")

"""## Limpieza de RAM"""

import gc
gc.collect()

# Muestra una lista simple con los tipos de datos, antes de empezar los groupby
print(df_licencias_optimizada.dtypes)

"""# Evolución y comportamiento institucional de las licencias médicas

## ¿Cómo ha evolucionado la emisión y autorización de licencias a lo largo de los años?
"""

# Extraemos el Año y el Trimestre para poder ordenar correctamente
df_licencias_optimizada['Anio'] = df_licencias_optimizada['FechaInformacion'].str[:4]
df_licencias_optimizada['Trimestre'] = df_licencias_optimizada['FechaInformacion'].str[-1]

# GroupBy de Año por RunTrabajador y NumeroDiasAutorizados
evolucion_anual = df_licencias_optimizada.groupby('Anio').agg(
    Total_Licencias=('RunTrabajador', 'count'),
    Promedio_Dias_Autorizados=('NumeroDiasAutorizados', 'mean')
).reset_index()

# Visualización de la evolución
print("--- Evolución Anual de Licencias ---")
display(evolucion_anual)

# Configuración del estilo
sns.set_theme(style="whitegrid")

# Crear una figura con 2 gráficos
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), sharex=True)

# Paleta de colores
color_licencias = '#2b5c8f'  # Azul corporativo
color_dias = '#d95f02'       # Naranja/Óxido


# GRÁFICO 1: Evolución del Total de Licencias
sns.lineplot(
    data=evolucion_anual, x='Anio', y='Total_Licencias',
    ax=ax1, color=color_licencias, linewidth=3, marker='o', markersize=8
)
ax1.set_title('Evolución Temporal de Emisión de Licencias Médicas', fontsize=14, fontweight='bold', pad=15, color='#333333')
ax1.set_ylabel('Total de Licencias', fontsize=11, fontweight='bold', color=color_licencias)
ax1.tick_params(axis='y', labelcolor=color_licencias)

# Formatear el eje Y en millones (M) o miles (K)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-6:.1f}M' if x >= 1e6 else f'{x*1e-3:.0f}K' if x >= 1e3 else f'{x:.0f}'))

# Añadir etiquetas de datos en los puntos clave de las licencias
for x, y in zip(evolucion_anual['Anio'], evolucion_anual['Total_Licencias']):
    texto_l = f'{y*1e-6:.2f}M' if y >= 1e6 else f'{y*1e-3:.0f}K'
    ax1.annotate(texto_l, xy=(x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, fontweight='bold', color=color_licencias)

# GRÁFICO 2: Evolución de la Duración Promedio
sns.lineplot(
    data=evolucion_anual, x='Anio', y='Promedio_Dias_Autorizados',
    ax=ax2, color=color_dias, linewidth=3, marker='s', markersize=8
)
ax2.set_title('Evolución de la Duración Promedio de Días Autorizados', fontsize=14, fontweight='bold', pad=15, color='#333333')
ax2.set_xlabel('Año', fontsize=12, fontweight='bold', labelpad=10)
ax2.set_ylabel('Promedio de Días', fontsize=11, fontweight='bold', color=color_dias)
ax2.tick_params(axis='y', labelcolor=color_dias)

# Asegurar que se marquen todos los años en el eje X
ax2.set_xticks(evolucion_anual['Anio'])

# Añadir etiquetas de datos con los días exactos
for x, y in zip(evolucion_anual['Anio'], evolucion_anual['Promedio_Dias_Autorizados']):
    ax2.annotate(f'{y:.1f} días', xy=(x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, fontweight='bold', color=color_dias)

# Limpieza de bordes y ajuste de espacio
sns.despine()
plt.tight_layout()

# Mostrar gráficos
plt.show()

"""## ¿Existe un aumento estacional en ciertos trimestres?"""

# Agrupar datos por Año y Trimestre
estacionalidad = df_licencias_optimizada.groupby(['Anio', 'Trimestre']).agg(
    Total_Licencias=('RunTrabajador', 'count'),
    Promedio_Dias=('NumeroDiasAutorizados', 'mean')
).reset_index()

# Mostrar gráfico
print("--- Aumento estacional ---")
display(estacionalidad)

# Crear una columna de texto combinada para el eje X (ej: "2024-T1")
estacionalidad['Periodo'] = estacionalidad['Anio'].astype(str) + '-' + estacionalidad['Trimestre'].astype(str)

# Configurar el gráfico
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 6))

# Dibujar la evolución trimestral
ax = sns.lineplot(
    data=estacionalidad,
    x='Periodo',
    y='Total_Licencias',
    marker='o',
    linewidth=2.5,
    color='#2b5c8f'
)

# Estética y títulos
plt.title('Análisis Estacional: Volumen de Licencias Médicas por Trimestre', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Periodo (Año-Trimestre)', fontsize=11)
plt.ylabel('Cantidad de Licencias', fontsize=11)
plt.xticks(rotation=45)

# Formatear el eje Y de manera ejecutiva (en miles o millones)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-6:.1f}M' if x >= 1e6 else f'{x*1e-3:.0f}K' if x >= 1e3 else f'{x:.0f}'))

# Ajustar bordes
sns.despine()
plt.tight_layout()

#Mostrar gráfico
plt.show()

"""# Analizar la distribución de licencias por género, grupo etario y condición laboral

## ¿Qué diferencias existen entre hombres y mujeres en cantidad de licencias, duración y subsidio recibido?
"""

dif_sexo = df_licencias_optimizada.groupby('SexoTrabajador').agg(
    Cantidad_Licencias=('RunTrabajador', 'count'),
    Duracion_Promedio_Dias=('NumeroDias', 'mean'),
    Duracion_Total_Dias=('NumeroDias', 'sum'),
    Subsidio_Promedio=('MontoSubsidioLiquido', 'mean'),
    Subsidio_Total=('MontoSubsidioLiquido', 'sum')
).reset_index()

print("Resumen de Licencias por Sexo")
print(dif_sexo)

# Modificar el estilo estético global
sns.set_theme(style="whitegrid")

# Crear una figura con 3 subgráficos lado a lado
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Definir una paleta de colores
paleta = sns.color_palette("Set2")

# GRÁFICO 1: Cantidad de Licencias
sns.barplot(
    data=dif_sexo,
    x='SexoTrabajador',
    y='Cantidad_Licencias',
    ax=axes[0],
    palette=paleta,
    hue='SexoTrabajador',
    legend=False
)
axes[0].set_title('Cantidad de Licencias por Sexo', fontsize=13, fontweight='bold', pad=15)
axes[0].set_xlabel('Sexo', fontsize=11)
axes[0].set_ylabel('Total de Licencias', fontsize=11)

# Formatear el eje Y
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-6:.1f}M' if x >= 1e6 else f'{x*1e-3:.0f}K' if x >= 1e3 else f'{x:.0f}'))

for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height():,.0f}",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center',
                     xytext=(0, 8),
                     textcoords='offset points', fontsize=10, fontweight='bold', color='#444444')


# GRÁFICO 2: Duración Promedio en Días
sns.barplot(
    data=dif_sexo,
    x='SexoTrabajador',
    y='Duracion_Promedio_Dias',
    ax=axes[1],
    palette=paleta,
    hue='SexoTrabajador',
    legend=False
)
axes[1].set_title('Duración Promedio de Licencias', fontsize=13, fontweight='bold', pad=15)
axes[1].set_xlabel('Sexo', fontsize=11)
axes[1].set_ylabel('Días promedio', fontsize=11)

# Añadir etiquetas con el valor exacto encima de cada barra
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():.1f} días",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center',
                     xytext=(0, 8),
                     textcoords='offset points', fontsize=10, fontweight='bold', color='#444444')


# GRÁFICO 3: Subsidio Promedio Recibido
sns.barplot(
    data=dif_sexo,
    x='SexoTrabajador',
    y='Subsidio_Promedio',
    ax=axes[2],
    palette=paleta,
    hue='SexoTrabajador',
    legend=False
)
axes[2].set_title('Monto Subsidio Promedio ($)', fontsize=13, fontweight='bold', pad=15)
axes[2].set_xlabel('Sexo', fontsize=11)
axes[2].set_ylabel('Monto promedio líquido ($)', fontsize=11)

# Formatear el eje Y en miles para que sea legible
axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x*1e-3:.0f}K' if x >= 1e3 else f'${x:.0f}'))

# Añadir etiquetas con el valor en dinero encima de cada barra
for p in axes[2].patches:
    axes[2].annotate(f"${p.get_height():,.0f}",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center',
                     xytext=(0, 8),
                     textcoords='offset points', fontsize=10, fontweight='bold', color='#444444')


# Remover bordes superiores e derechos innecesarios en todos los gráficos
sns.despine()

# Ajustar el espacio entre los subgráficos automáticamente
plt.tight_layout()

# Mostrar gráfico
plt.show()

"""## ¿Qué grupo etario solicita más licencias?"""

dif_etario = df_licencias_optimizada.groupby('GrupoEtario').agg(
    Cantidad_Licencias=('RunTrabajador', 'count')
).reset_index()

# Ordenamos de mayor a menor para identificar fácilmente cuál grupo solicita más
grupo_mas_licencias = dif_etario.sort_values(by='Cantidad_Licencias', ascending=False)

print("Resumen de Licencias por Grupo Etario")
print(grupo_mas_licencias)

import plotly.express as px

# 1. Filtramos y ORDENAMOS de menor a mayor
# Usamos ascending=True para que en el gráfico horizontal la categoría
# más grande quede en la parte superior (posición más prominente).
grupo_mas_licencias_filtrado = grupo_mas_licencias[
    ~grupo_mas_licencias["GrupoEtario"].isin(["Niñez", "Sin información"])
].sort_values(by="Cantidad_Licencias", ascending=True)

# 2. Configuración del gráfico de barras horizontales
fig = px.bar(
    grupo_mas_licencias_filtrado,
    x="Cantidad_Licencias",
    y="GrupoEtario",
    orientation='h', # Define que sea horizontal
    title="<b>Participación de Licencias Médicas por Grupo Etario</b>",
    template="plotly_white",
    color="Cantidad_Licencias", # Usamos el valor para el degradé
    color_continuous_scale=px.colors.sequential.Blues_r # Mantenemos tu paleta
)

# 3. Personalización de etiquetas y formato
fig.update_traces(
    texttemplate='%{x:,.0f}', # Formato numérico con separador de miles
    textposition='outside',   # Etiquetas fuera de la barra para limpieza
    hovertemplate=(
        "<b>Grupo Etario:</b> %{y}"
        "<br><b>Licencias:</b> %{x:,.0f}"
        "<extra></extra>"
    )
)

# 4. Diseño final ejecutivo
fig.update_layout(
    title_font_size=16,
    xaxis_title="Cantidad de Licencias",
    yaxis_title="Grupo Etario",
    coloraxis_showscale=False, # Ocultamos la barra de color lateral para mayor limpieza
    margin=dict(l=100, r=20, t=50, b=50) # Margen izquierdo extra para los nombres
)

# Mostrar gráfico
fig.show()

"""# Estudio específico según calidad del trabajador

## ¿Cómo varía la frecuencia de licencias según el tipo de trabajador?
"""

frec_tipo_trabajador = df_licencias_optimizada.groupby('CalidadTrabajador').agg(
    Frecuencia_Licencias=('RunTrabajador', 'count'),       # Cuenta cuántas licencias hay por categoría
).reset_index()

frec_tipo_trabajador = frec_tipo_trabajador.sort_values(by='Frecuencia_Licencias', ascending=False)

print("Frecuencia de Licencias por Tipo de Trabajador")
print(frec_tipo_trabajador)

# Gráfico
fig = px.bar(
    frec_tipo_trabajador,
    x="Frecuencia_Licencias",
    y="CalidadTrabajador",
    orientation="h",
    color="Frecuencia_Licencias",
    color_continuous_scale="Tealgrn",
    text="Frecuencia_Licencias",
    title="<b>Frecuencia de Licencias por Calidad del Trabajador</b>"
)

# Personalización para etiquetas y información
fig.update_traces(
    texttemplate='%{text:,.0f}',
    textposition='outside'
)

# Diseño general
fig.update_layout(
    xaxis_title="Cantidad de Licencias",
    yaxis_title="Calidad del Trabajador",
    coloraxis_showscale=False,
    template="plotly_white"
)

# Mostrar gráfico
fig.show()

"""## ¿Qué diferencias hay entre trabajadores públicos afectos y no afectos a la Ley 18.834 con el resto?"""

resumen_ley = df_licencias_optimizada.groupby('Clasificacion_Ley').agg(
    Cantidad_Licencias=('RunTrabajador', 'count'),
    Duracion_Promedio_Dias=('NumeroDias', 'mean'),
    Subsidio_Promedio_Liquido=('MontoSubsidioLiquido', 'mean')
).reset_index()

print("Comparación Global: Afectos vs No Afectos vs Resto")
print(resumen_ley)

# Gráfico
px.bar(
    resumen_ley,
    x="Clasificacion_Ley",
    y="Cantidad_Licencias",
    color="Clasificacion_Ley",
    text="Cantidad_Licencias"
)

"""# Detección de anomalías y patrones atípicos

## ¿Qué trabajadores (RunTrabajador) reciben licencias repetidas o con altos días acumulados?
"""

# Agrupar por RunTrabajador para obtener el total de NumeroDias por cada uno
dias_por_trabajador = df_licencias_optimizada.groupby('RunTrabajador')['NumeroDias'].sum().reset_index().sort_values(by= 'NumeroDias',ascending = False)

# Calcular el percentil 95 sobre ese total agrupado
quantile_95_agrupado = dias_por_trabajador['NumeroDias'].quantile(0.95)
print(f"El percentil 95 de NumeroDias (total por trabajador) es: {quantile_95_agrupado}")

ruts_sobre_quantile = dias_por_trabajador[dias_por_trabajador['NumeroDias'] > quantile_95_agrupado]
ruts_sobre_quantile

# Gráfico
fig = px.histogram(
    dias_por_trabajador,
    x="NumeroDias",
    nbins=50,
    title="<b>Distribución de Días Acumulados de Licencia por Trabajador</b>",
    template="plotly_white"
)

# Visualización del percentil 95
fig.add_vline(
    x=quantile_95_agrupado,
    line_dash="dash",
    line_color="red",
    annotation_text="Percentil 95"
)

# Formatear el eje Y a escala logarítmica
fig.update_yaxes(type="log")

#Mostrar gráfico
fig.show()

"""## ¿Qué profesionales (RunProfesional) emiten un número anómalo de licencias?"""

import plotly.express as px

# 1. Agrupar por RunProfesional para contar las licencias
licencias_por_profesional = df_licencias_optimizada.groupby('RunProfesional').size().reset_index(name='Cantidad_Licencias')

# 2. Calcular el P95
p95 = licencias_por_profesional['Cantidad_Licencias'].quantile(0.95)

vibrant_colors = {
    'Comportamiento Normal': '#7f7f7f',  # Gris más oscuro
    'Paciente Crónico': '#1f77b4',      # Azul intenso
    'Alta Renta / Anomalía': '#ff7f0e', # Naranja brillante
    'Caso Extremo Absoluto': '#d62728'  # Rojo vibrante
}

# 3. Crear el gráfico de dispersión
# Usamos un índice como eje X para distribuir los puntos
fig = px.scatter(
    licencias_por_profesional,
    x=licencias_por_profesional.index,
    y='Cantidad_Licencias',
    title=f"<b>Dispersión de Emisión de Licencias por Profesional</b><br><sup> El P95 ({p95:.0f} licencias)</sup>",
    template="plotly_white",
    color='Cantidad_Licencias',
    color_discrete_map=vibrant_colors,
    hover_data=['RunProfesional']
)


# 5. Ajustes
fig.update_layout(
    xaxis_title="Índice de Profesionales",
    yaxis_title="Cantidad de Licencias",
    showlegend=False
)

fig.show()

"""## ¿Existen patrones de licencias con valores atípicos por duración o subsidio?"""

p95_dias = df_licencias_optimizada["NumeroDias"].quantile(0.95)

licencias_largas = df_licencias_optimizada[
    df_licencias_optimizada["NumeroDias"] > p95_dias
]

p95_subsidio = df_licencias_optimizada["MontoSubsidioLiquido"].quantile(0.95)

subsidios_altos = df_licencias_optimizada[
    df_licencias_optimizada["MontoSubsidioLiquido"] > p95_subsidio
]

print(p95_dias)
print(p95_subsidio)
licencias_largas.head()

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Crear la estructura de dos subplots en paralelo
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>Distribución de Duración (Días por Licencia)</b>",
        "<b>Distribución de Costo (Subsidio por Licencia)</b>"
    ),
    horizontal_spacing=0.12
)

# Histograma de NumeroDias
fig.add_trace(
    go.Histogram(
        x=df_licencias_optimizada['NumeroDias'],
        nbinsx=50,
        name='Duración',
        marker_color='#2b5c8f',  # Azul corporativo uniforme
        hovertemplate="<b>Rango de Días:</b> %{x}<br><b>Cantidad Licencias:</b> %{y:,.0f}<extra></extra>"
    ),
    row=1, col=1
)

# Agregar línea vertical para el Percentil 95 de Días
fig.add_vline(
    x=p95_dias,
    line_dash="dash",
    line_color="#d95f02",  # Color óxido/alerta para contraste
    line_width=2,
    annotation_text=f"<b>P95 Días:</b> {p95_dias:.0f} días",
    annotation_position="top right",
    annotation_font=dict(color="#d95f02", size=10),
    row=1, col=1
)

# Histograma de MontoSubsidioLiquido
fig.add_trace(
    go.Histogram(
        x=df_licencias_optimizada['MontoSubsidioLiquido'],
        nbinsx=50,
        name='Subsidio',
        marker_color='#4a7c59',  # Verde contable elegante
        hovertemplate="<b>Rango Subsidio:</b> %{x}<br><b>Cantidad Licencias:</b> %{y:,.0f}<extra></extra>"
    ),
    row=1, col=2
)

# Agregar línea vertical para el Percentil 95 de Subsidio
fig.add_vline(
    x=p95_subsidio,
    line_dash="dash",
    line_color="#d95f02",
    line_width=2,
    annotation_text=f"<b>P95 Subsidio:</b> ${p95_subsidio*1e-6:,.1f}M".replace(',', '.'),
    annotation_position="top right",
    annotation_font=dict(color="#d95f02", size=10),
    row=1, col=2
)

# Ajustes generales de diseño ejecutivo y compresión de escalas
fig.update_layout(
    title_text="<b>Auditoría de Extremos: Umbrales del Percentil 95 por Licencia Individual</b>",
    title_font_size=16,
    template="plotly_white",
    height=550,
    showlegend=False,  # Ocultamos la leyenda ya que los títulos de paneles aclaran las métricas
    hoverlabel=dict(bgcolor="white", font_size=12)
)

# Formatear ejes Y con separadores de miles estándar chilenos
fig.update_yaxes(tickformat=",d", row=1, col=1)
fig.update_yaxes(tickformat=",d", row=1, col=2)

# Formatear el eje X financiero dividirá de forma automática y mostrará abreviaturas compactas (K o M)
fig.update_xaxes(tickformat="$", tickprefix="$", row=1, col=2)

# Mostrar gráficos
fig.show()

"""## ¿Existen patrones de licencias con valores atípicos por duración o subsidio? (Por tipo de licencia)"""

# Agrupar por CategoriaDiagnostico para obtener el total de NumeroDias
dias_por_categoria = df_licencias_optimizada.groupby('CategoriaDiagnostico')['NumeroDias'].sum().reset_index().sort_values(by = 'NumeroDias', ascending = False)

# Calcular el percentil 95 sobre el total de días por categoría
quantile_95_categoria = dias_por_categoria['NumeroDias'].quantile(0.95)
print(f"El percentil 95 de NumeroDias (total por categoría) es: {quantile_95_categoria}")

# Identificar y filtrar las categorías que superan el umbral (Top 10)
resumen_categorias_top = dias_por_categoria[dias_por_categoria['NumeroDias'] > quantile_95_categoria].sort_values(by='NumeroDias', ascending=False).head()

print("Categorías de diagnóstico sobre el percentil 95:")
display(resumen_categorias_top)

# Gráfico de barras horizontales
fig = px.bar(
    resumen_categorias_top,
    x="NumeroDias",
    y="CategoriaDiagnostico",
    orientation="h",
    color="NumeroDias",
    color_continuous_scale="Tealgrn",
    text="NumeroDias",
    title="<b>Categorías de Diagnóstico sobre el Percentil 95 de Días Acumulados</b>",
    template="plotly_white"
)

# Etiquetas y información
fig.update_traces(
    texttemplate='%{text:,.0f}',
    textposition='outside'
)

# Modificar el diseño general
fig.update_layout(
    xaxis_title="Total de días acumulados de licencia",
    yaxis_title="Categoría de diagnóstico",
    coloraxis_showscale=False,
    yaxis=dict(categoryorder="total ascending"),
    title_x=0.5
)

# Definir linea percentil 95
fig.add_vline(
    x=quantile_95_categoria,
    line_dash="dash",
    line_color="red",
    annotation_text="Percentil 95",
    annotation_position="top"
)

# Mostrar gráfico
fig.show()

# Agrupar correctamente incluyendo cuenta de licencias y suma de días
dias_por_categoria = df_licencias_optimizada.groupby('CategoriaDiagnostico').agg(
    Dias_Acumulados=('NumeroDias', 'sum')
).reset_index()

# Calcular los Tops
top_dias_acumuladoss = dias_por_categoria.sort_values(by='Dias_Acumulados', ascending=False).head(10)

print("Top 10 Enfermedades: Mayor cantidad de licencias")
display(top_dias_acumuladoss)

# Gráfico
fig = px.treemap(
    top_dias_acumuladoss,
    path=["CategoriaDiagnostico"],
    values="Dias_Acumulados",
    color="Dias_Acumulados",
    color_continuous_scale="Tealgrn",
    title="<b>Top 10 Categorías de Diagnóstico por Días Acumulados</b>"
)

# Mostrar gráfico
fig.show()
