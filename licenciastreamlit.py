# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# **Unión de Trimestres**
import os
import pandas as pd
import gc
# ==========================================
# AJUSTE DE DISEÑO (Quitar espacio en blanco superior)
# ==========================================
st.markdown(
    """
    <style>
    /* 1. Eliminar el espacio (padding) del contenedor principal */
    .block-container {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    
    /* 2. Hacer transparente el encabezado de Streamlit para que no ocupe espacio visual */
    header {
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
@st.cache_data
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

    columnas_finales = ['CodigoIsapre', 'FechaInformacion', 'RunTrabajador', 'NumeroDias',
                        'EdadTrabajador', 'SexoTrabajador', 'ActividadLaboral', 'TipoLicencia',
                        'CaracteristicasReposo', 'RunProfesional', 'TipoProfesional',
                        'TipoLicenciaContraloria', 'NumeroDiasAutorizados', 'DiagnosticoPrincipal',
                        'TipoResolucion', 'Periodo', 'Region', 'CalidadTrabajador',
                        'EntidadPagadora', 'NumeroDiasPagar', 'MontoSubsidioLiquido',
                        'OtrosDiagnosticos', 'CausaRechazoModificacion', 'FechaContratoTrabajo',
                        'MontoBaseCalculoSubsidio', 'RutEmpleador', 'CategoriaDiagnostico',
                        'GrupoEtario', 'Clasificacion_Ley','Isapres']

    return df[[c for c in columnas_finales if c in df.columns]]

   # 3. y 4. CICLO DE CARGA Y UNIÓN (SÚPER OPTIMIZADO PARA MEMORIA)
@st.cache_resource
def cargar_toda_la_data():
    ruta_base = r"C:\Users\arena\OneDrive - Corporación Santo Tomas\7mo semestre\MACHINE LEARNING\PROYECTO\CARPETAS ZIP"
    lista_dfs = []

    # 1. Definir SOLO las columnas que se usan en tu procesar_y_limpiar y gráficos
    columnas_a_leer = [
        'CodigoIsapre', 'FechaInformacion', 'RunTrabajador', 'NumeroDias',
        'EdadTrabajador', 'SexoTrabajador', 'ActividadLaboral', 'TipoLicencia',
        'CaracteristicasReposo', 'RunProfesional', 'TipoProfesional',
        'TipoLicenciaContraloria', 'NumeroDiasAutorizados', 'DiagnosticoPrincipal',
        'TipoResolucion', 'Periodo', 'Region', 'CalidadTrabajador',
        'EntidadPagadora', 'NumeroDiasPagar', 'MontoSubsidioLiquido',
        'OtrosDiagnosticos', 'CausaRechazoModificacion', 'FechaContratoTrabajo',
        'MontoBaseCalculoSubsidio', 'RutEmpleador'
    ]

    # 2. Asignar tipos de datos eficientes
    # Convertir textos a 'category' y números a 'float32' reduce drásticamente el peso
    # 2. Asignar tipos de datos eficientes
    tipos_optimizados = {
        'SexoTrabajador': 'category',
        'Region': 'category',
        'TipoLicencia': 'category',
        'CalidadTrabajador': 'category',
        'EntidadPagadora': 'category',
        'EdadTrabajador': 'category',  
        'NumeroDias': 'float32',
        'NumeroDiasAutorizados': 'float32',
        'MontoSubsidioLiquido': 'float32'
    }

    for archivo in os.listdir(ruta_base):
        if archivo.endswith(".txt"):
            print(f"Leyendo optimizado: {archivo}")
            # Leer el archivo aplicando los filtros desde la raíz
            df_temp = pd.read_csv(
                os.path.join(ruta_base, archivo), 
                sep="|", 
                header=None, 
                names=columnas_licencias, # Tu lista original de 45 columnas para mapear
                usecols=columnas_a_leer,  # Solo carga a la RAM estas columnas
                dtype=tipos_optimizados,  # Comprime la memoria
                encoding="latin-1", 
                on_bad_lines="skip"
            )

            # Aplicar la función de limpieza
            lista_dfs.append(procesar_y_limpiar(df_temp))

            # Liberar memoria de inmediato
            del df_temp
            gc.collect()

    # Unión final
    df_final = pd.concat(lista_dfs, ignore_index=True)
    df_final.drop_duplicates(inplace=True)
    
    # Preparación de fechas
    df_final['Anio'] = df_final['FechaInformacion'].astype(str).str[:4]
    df_final['Trimestre'] = df_final['FechaInformacion'].astype(str).str[-1]
    
    return df_final

# ==========================================
# EJECUCIÓN DEL CACHÉ (La magia ocurre aquí)
# ==========================================
# Esta línea solo tomará tiempo la PRIMERA vez que abras la app.
# Luego, cada vez que uses un filtro, la carga será instantánea.
with st.spinner('Cargando y procesando base de datos...'):
    df_licencias_optimizada = cargar_toda_la_data()

#FIN DE UNIÓN Y LIMPIEZA




# ==========================================
# 1. PREPARACIÓN DE FECHAS
# ==========================================
# Aseguramos que sea texto y extraemos los primeros 4 caracteres para el Año
df_licencias_optimizada['Anio'] = df_licencias_optimizada['FechaInformacion'].astype(str).str[:4]

# Extraemos el último caracter para el Trimestre
df_licencias_optimizada['Trimestre'] = df_licencias_optimizada['FechaInformacion'].astype(str).str[-1]


# ==========================================
# 2. MENÚ LATERAL DE FILTROS (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Filtros del Panel")

opciones_etarias = sorted(df_licencias_optimizada['GrupoEtario'].unique().tolist())

# Usamos multiselect para permitir varias opciones a la vez
lista_completa = df_licencias_optimizada['GrupoEtario'].unique().tolist()

# 2. Definimos cuáles queremos excluir de la vista
excluidos = ["Niñez", "Sin información"]

# 3. Creamos la lista final filtrada (usando una "comprensión de lista")
opciones_etarias = sorted([x for x in lista_completa if x not in excluidos])

# Ahora las listas sí encontrarán la columna 'Anio'
opciones_anio = ["Todos"] + list(df_licencias_optimizada['Anio'].unique())
opciones_trimestre = ["Todos"] + list(df_licencias_optimizada['Trimestre'].unique())
opciones_sexo = ["Todos"] + list(df_licencias_optimizada['SexoTrabajador'].unique())

# Crear los controles
filtro_anio = st.sidebar.selectbox("Selecciona el Año", opciones_anio)
filtro_trimestre = st.sidebar.selectbox("Selecciona el Trimestre", opciones_trimestre)
filtro_sexo = st.sidebar.selectbox("Selecciona el Sexo", opciones_sexo)
# Definir opciones excluyendo lo que no quieres
opciones_etarias = sorted([x for x in df_licencias_optimizada['GrupoEtario'].unique() if x not in ["Niñez", "Sin información"]])

# Este filtro NO altera df_filtrado, solo guarda la selección en 'filtro_etario'
filtro_etario = st.sidebar.multiselect(
    "Filtrar solo Gráfico Etario",
    options=opciones_etarias,
    default=opciones_etarias,
    key="filtro_etario_exclusivo"
)
# 3. Aplicar los filtros a la base de datos

df_filtrado = df_licencias_optimizada

if filtro_anio != "Todos":
    df_filtrado = df_filtrado[df_filtrado['Anio'] == filtro_anio]

if filtro_trimestre != "Todos":
    df_filtrado = df_filtrado[df_filtrado['Trimestre'] == filtro_trimestre]

if filtro_sexo != "Todos":
    df_filtrado = df_filtrado[df_filtrado['SexoTrabajador'] == filtro_sexo]
# --- FILTRO MÚLTIPLE DE GRUPO ETARIO ---





# Mostrar resumen del filtro

#----------------------------------------------------------------------------------------------------------------------
# ==========================================
# LOGOS INSTITUCIONALES (IZQUIERDA Y DERECHA)
# ==========================================
# Creamos 3 columnas: [Izquierda (2), Centro vacío (6), Derecha (2)]
col_izq, col_centro, col_der = st.columns([2, 6, 2])

with col_izq:
    # Aquí va tu primer logo (Superintendencia de Salud)
    ruta_logo_salud = r"C:\Users\arena\Desktop\ProyectoLicencias\logo.png"
    st.image(ruta_logo_salud, use_container_width=True)

# La columna del centro (col_centro) la dejamos sin código para que quede vacía

with col_der:
    # Aquí va el nuevo logo de la UST
    ruta_logo_ust = r"C:\Users\arena\Desktop\ProyectoLicencias\ustlogo.png"
    st.image(ruta_logo_ust, use_container_width=True)

# ==========================================
# BANNER PRINCIPAL
# ==========================================
ruta_banner = r"C:\Users\arena\Desktop\ProyectoLicencias\banner.png"
st.image(ruta_banner, use_container_width=True)
#-----------------------------------------------------------------------------------------------------------------------






# ==========================================
# 4. INDICADORES DINÁMICOS
# ==========================================
st.header("**Indicadores Principales**")

# Calculamos los totales usando el dataframe que ya pasó por los filtros
total_licencias = len(df_filtrado)
monto_subsidio = df_filtrado['MontoSubsidioLiquido'].sum()
total_dias_autorizados = df_filtrado['NumeroDiasAutorizados'].sum()


# Usamos una validación para evitar errores si un filtro deja 0 resultados
if not df_filtrado.empty:
    region_frecuente = df_filtrado['Region'].value_counts().idxmax()
    isapre_frecuente = df_filtrado['Isapres'].value_counts().idxmax()
else:
    region_frecuente = "Sin datos"
    isapre_frecuente = "Sin datos"

col1, col2, col3, col4, col5 = st.columns([1.2, 1.6, 1.3, 0.7, 1.0])

with col1:
    st.metric(
        label="Total de Licencias",
        value=f"{total_licencias:,.0f}".replace(",", ".")
    )

with col2:
    # Si el monto es mayor o igual a 1.000 millones, lo resumimos
    if monto_subsidio >= 1000000000:
        monto_calculado = monto_subsidio / 1000000
        texto_monto = f"${monto_calculado:,.0f} MM".replace(",", ".")
    
    # Si el monto es menor (como tus 17 millones), lo mostramos completo
    else:
        texto_monto = f"${monto_subsidio:,.0f}".replace(",", ".")
        
    st.metric(
        label="Monto Subsidio Total",
        value=texto_monto
    )

with col3:
    st.metric(
        label="Días Autorizados Totales",
        value=f"{total_dias_autorizados:,.0f}".replace(",", ".")
    )

with col4:
    # Como la región es un número, str() asegura que Streamlit lo muestre correctamente (ej: "13")
    st.metric(
        label="Región principal",
        value=str(region_frecuente)
    )

with col5:
    st.metric(
        label="Isapre principal",
        value=str(isapre_frecuente)
    )





st.markdown("---")

# El número 1.5 hace que la columna del medio sea un 50% más ancha que las otras dos
col1, col2, col3 = st.columns([1, 1.5, 1])
st.set_page_config(layout="wide")


# ==========================================
# 5. PANEL DE GRÁFICOS ORGANIZADO POR PESTAÑAS
# ==========================================

# 1. Creamos las pestañas con nombres precisos
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "👨‍⚕️Presentación",
    "📊Fuentes de Datos",
    "📈 Evolución Institucional", 
    "👥 Distribución Demográfica", 
    "💼 Calidad del Trabajador", 
    "🚨 Detección de Anomalías"
    ])

    # ---------------------------------------------------------
    # PESTAÑA 1: Evolución
    # ---------------------------------------------------------

with tab1:
    st.header("🎯 Objetivo del Proyecto")
    
    # st.markdown permite escribir texto con negritas, listas y emojis
    st.markdown("""
    Bienvenidos al **Panel de Inteligencia de Negocios para el Análisis de Licencias Médicas**.
    
    Este proyecto, desarrollado para la **Universidad Santo Tomás**, tiene como objetivo principal proporcionar una herramienta visual, interactiva y analítica de apoyo a la gestión de la **Superintendencia de Salud**.
    
    **¿Qué puedes hacer en este panel?**
    * 📊 **Monitorear** el volumen total y el costo económico (subsidios) de las licencias médicas aprobadas.
    * 🗺️ **Analizar** la distribución demográfica, temporal y geográfica a nivel nacional.
    * 🔍 **Detectar** anomalías y comportamientos atípicos en la emisión o recepción de licencias.
    
    *Utiliza el menú lateral izquierdo para filtrar los datos por Año, Trimestre, Sexo y Grupo Etario.*
    """)
    
    st.success("Para comenzar, navega por las pestañas superiores.")


with tab2:
    st.header("🗄️ Origen y Tratamiento de los Datos")
    
    # st.info crea un recuadro azul de información muy corporativo
    st.info("""🔒 **Privacidad:** Todos los datos presentados en este panel son de carácter público 
    """)
    
    st.markdown("""
    **Estructura de la Información:**
    * Los registros provienen de los consolidadores oficiales asociados a la **Superintendencia de Salud**.
    * El periodo analizado abarca el ciclo **2021 - 2026** en Trimestres.
    * La base de datos incluye variables cualitativas y cuantitativas como: Entidad pagadora (Isapres), Días de reposo, Montos de subsidio, Diagnósticos principales, etc.
    * Para revisar los datos se podrá descargar y enlaces:
    """)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.link_button("Nomenclatura de Diagnósticos", "https://ais.paho.org/classifications/chapters/pdf/volume1.pdf")
    with col2:
        st.link_button("Portal Nombre Isapres", "https://www.superdesalud.gob.cl/registro/isapres/")
    with col3:
        st.link_button("Portal Datos Abiertos", "https://www.superdesalud.gob.cl/tax-biblioteca-digital/datos-abiertos-de-isapres-6988/")
    with col4:
       def render_fuentes_de_datos(df):
        st.header("🗄️ Origen y Diccionario de Datos")
    # 1. Definimos la ruta del script actual para buscar el PDF en la misma carpeta
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    # Intentamos buscar el archivo. 
    # OJO: Si tu archivo se llama solo "diccionario", cámbialo abajo a "diccionario"
    nombre_archivo = "diccionario.pdf" 
    ruta_completa = os.path.join(directorio_actual, nombre_archivo)
    
    if os.path.exists(ruta_completa):
        with open(ruta_completa, "rb") as f:
            pdf_bytes = f.read()
            
        st.download_button(
            label="📥 Descargar Diccionario de Datos (PDF)",
            data=pdf_bytes,
            file_name=nombre_archivo,
            mime="application/pdf"
        )
    else:
        st.error(f"⚠️ No se pudo encontrar el archivo en: {ruta_completa}")
        st.info("Asegúrate de que el archivo se llame exactamente 'diccionario.pdf' y esté en la misma carpeta que 'licenciastreamlit.py'.")

    st.subheader("Vista previa de la Base de Datos")
    st.write("A continuación se muestran los primeros 100 registros según los filtros que tengas aplicados actualmente en el menú lateral:")
    
    # Esto genera una tabla interactiva nativa de Streamlit para auditar los datos
    st.dataframe(df_filtrado.head(100), use_container_width=True)



with tab3:
    st.header("Evolución y comportamiento institucional de las licencias médicas")
    st.caption("⚠️ *Nota de auditoría:* La caída drástica en el volumen del año 2026 (214K) no representa una anomalía de datos, sino que responde a que el dataset contiene registros parciales del año en curso.")
    st.markdown("""
    Esta sección permite analizar el comportamiento institucional de las licencias médicas en Chile durante el período 2021–2026, 
    mediante indicadores que reflejan su evolución temporal, distribución y principales características.
    """)
    
    st.subheader("¿Cómo ha evolucionado la emisión y autorización a lo largo de los años?")

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
    print(evolucion_anual)

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
    st.pyplot(fig)
    st.info("""**Análisis de Correlación Inversa: Volumen vs. Severidad**""")
    st.markdown("""
    Tras el peak de 2.09 millones de licencias en 2022, el volumen cayó sostenidamente, mientras que el reposo promedio 
    aumentó de 8.1 a 11.0 días hacia 2026. Esto demuestra que la fiscalización ha mitigado con éxito las licencias de 
    corta duración, concentrando los recursos del sistema en patologías más complejas y crónicas
    """)
    st.markdown("---")
    


    st.subheader("¿Existe un aumento estacional en ciertos trimestres?")
# --- GRÁFICO ESTACIONAL (Ahora interactivo con tus filtros actuales) ---

# 1. Usamos df_filtrado (que ya contiene los filtros de Año, Trimestre y Sexo)
# Esto hará que el gráfico se actualice solo cuando cambies cualquier filtro del sidebar
    estacionalidad = df_filtrado.groupby(['Anio', 'Trimestre']).agg(
    Total_Licencias=('RunTrabajador', 'count')
    ).reset_index()

# 2. Crear la columna de periodo para el eje X
    estacionalidad['Periodo'] = estacionalidad['Anio'].astype(str) + '-' + estacionalidad['Trimestre'].astype(str)

# 3. Dibujar el gráfico
    sns.set_theme(style="whitegrid")
    fig2 = plt.figure(figsize=(14, 6))

    ax = sns.lineplot(
    data=estacionalidad,
    x='Periodo',
    y='Total_Licencias',
    marker='o',
    linewidth=2.5,
    color='#2b5c8f'
)

# Títulos y formato
    plt.title('Análisis Estacional: Volumen de Licencias Médicas por Trimestre', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Periodo (Año-Trimestre)', fontsize=11)
    plt.ylabel('Cantidad de Licencias', fontsize=11)
    plt.xticks(rotation=45)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-6:.1f}M' if x >= 1e6 else f'{x*1e-3:.0f}K' if x >= 1e3 else f'{x:.0f}'))

    sns.despine()
    plt.tight_layout()

# Mostrar en Streamlit
    st.pyplot(fig2)
    
    st.info("""**Análisis de Volumen por Periodos**""")
    st.markdown("""
    Los segundos trimestres de cada año (coincidentes con otoño e invierno) registran sistemáticamente los peaks 
    de emisión de licencias médicas, mientras que los primeros trimestres representan los valles de menor actividad.
    """)


    # ---------------------------------------------------------
    # PESTAÑA 2: Demografía
    # ---------------------------------------------------------
with tab4:
    st.header("Análisis de distribución por género y grupo etario")
    st.markdown("""
    Se identificará cómo se distribuyen las licencias médicas entre hombres y mujeres, así como entre los distintos 
    grupos de edad, para identificar patrones demográficos relevantes.
    """)
    st.subheader("¿Qué diferencias existen entre hombres y mujeres?")

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
    st.pyplot(fig)  

    st.info("""**Análisis por Género**""")
    st.markdown("""
    Registran un volumen sustancialmente mayor de licencias, una duración promedio de reposo 
    más extensa y, consecuentemente, un costo por subsidio promedio significativamente superior. 
    Esta brecha acumulativa demuestra que el género femenino constituye el principal foco 
    de impacto operacional y financiero para el sistema previsional.
    """)

    st.markdown("---")


    st.subheader("¿Qué grupo etario solicita más licencias?")

   # 1. Crear copia LOCAL (No altera df_filtrado)
    df_etario_local = df_filtrado[df_filtrado['GrupoEtario'].isin(filtro_etario)]

# 2. Agrupación usando la variable local
    grupo_mas_licencias = df_etario_local.groupby('GrupoEtario')['RunTrabajador'].count().reset_index()
    grupo_mas_licencias.columns = ['GrupoEtario', 'Cantidad_Licencias']

# Ordenar
    grupo_mas_licencias_filtrado = grupo_mas_licencias.sort_values(by="Cantidad_Licencias", ascending=True)

# 3. Configuración del gráfico
    paleta_oceano = ["#0F172A", "#1E3A8A", "#0284C7", "#0D9488", "#2DD4BF"]

    fig = px.bar(
        grupo_mas_licencias_filtrado,
        x="Cantidad_Licencias",
        y="GrupoEtario",
        orientation='h',
        title="<b>Participación de Licencias Médicas por Grupo Etario</b>",
        template="plotly_white",
        color="GrupoEtario", 
        color_discrete_sequence=paleta_oceano
)

# 4. Personalización (Igual a la tuya)
    fig.update_traces(
        texttemplate='%{x:,.0f}',
        textposition='outside',
        hovertemplate="<b>Grupo Etario:</b> %{y}<br><b>Licencias:</b> %{x:,.0f}<extra></extra>"
)

    fig.update_layout(
        title_font_size=16,
        xaxis_title="Cantidad de Licencias",
        yaxis_title="Grupo Etario",
        showlegend=False,
        margin=dict(l=100, r=20, t=50, b=50)
)

# Mostrar gráfico
    st.plotly_chart(fig, use_container_width=True, key="grafico_grupo_etario")


    st.info("""**Análisis por Grupo Etario**""")
    st.markdown("""
    Los segmentos de "Adultos" y "Adultos Jóvenes", los cuales dominan casi la totalidad del registro. 
    Este comportamiento es coherente con la demografía del mercado laboral, ya que estos grupos representan la fuerza de trabajo activa
     y en plena etapa de productividad asimismo de reproducción. Por el contrario, los extremos de la vida laboral registran una participación 
    marginal: los "Jóvenes" apenas alcanzan las 178K emisiones y la "Tercera Edad" representa el volumen más bajo con solo 53K, 
    debido a su salida natural hacia los sistemas de pensiones.
    """)


    # ---------------------------------------------------------
    # PESTAÑA 3: Calidad del Trabajador
    # ---------------------------------------------------------
with tab5:
    st.header("Estudio específico según calidad del trabajador")

    st.markdown(""" Este apartado presenta un análisis de las licencias médicas según la calidad del trabajador, con el 
    propósito de identificar patrones y diferencias entre las distintas categorías laborales durante el período 2021–2026.
                """)
    
    st.subheader("¿Cómo varía la frecuencia de licencias según el tipo de trabajador?")

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
    st.plotly_chart(fig, use_container_width=True, key="grafico_calidad_trabajador")
    use_container_width=True

    st.info("""**Análisis Frecuencia por Calidad**""")
    st.markdown("""
    Se revela una profunda asimetría en la distribución de licencias médicas según la situación contractual, 
    concentrándose la carga de manera crítica en los trabajadores dependientes del sector privado con más de 
    5.17 millones de registros. En segundo orden de relevancia destacan los funcionarios públicos afectos a la 
    Ley N.° 18.834 con 2.11 millones de tramitaciones, mientras que el resto de los segmentos muestran una participación 
    marginal inferior a las 730 mil emisiones cada uno. Esta polarización evidencia que el impacto operativo y 
    financiero del sistema de Isapres se sostiene principalmente sobre la fuerza laboral privada dependiente además 
    la administración pública regulada.
    """)

    st.markdown("---")




    st.subheader("¿Qué diferencias hay entre trabajadores públicos afectos y no afectos con el resto?")

    resumen_ley = df_licencias_optimizada.groupby('Clasificacion_Ley').agg(
    Cantidad_Licencias=('RunTrabajador', 'count'),
    Duracion_Promedio_Dias=('NumeroDias', 'mean'),
    Subsidio_Promedio_Liquido=('MontoSubsidioLiquido', 'mean')
    ).reset_index()

    print("Comparación Global: Afectos vs No Afectos vs Resto")
    print(resumen_ley)

    # Gráfico
    # Gráfico - ¡Ahora sí asignado a una variable!
    fig_ley = px.bar(
    resumen_ley,
    x="Clasificacion_Ley",
    y="Cantidad_Licencias",
    color="Clasificacion_Ley",
    text="Cantidad_Licencias",
    title="<b>Comparación Global por Ley</b>",
    template="plotly_white"
    )

    # Y lo llamamos con esa misma variable
    st.plotly_chart(fig_ley, use_container_width=True, key="grafico_clasificacion_ley")
    
    st.info("""**Análisis de Frecuencia por Calidad**""")
    st.markdown(""" En el ámbito estatal, los funcionarios "Públicos Afectos" triplican en volumen a los "No Afectos", mientras que los registros 
    "Sin información" representan un residuo marginal de solo 104K. Esta disparidad confirma que el régimen laboral del sector 
    privado e independiente, seguido por la administración pública regulada, concentra la mayor carga operacional y financiera 
    del sistema previsional.
    """)


    # ---------------------------------------------------------
    # PESTAÑA 4: Anomalías
    # ---------------------------------------------------------
with tab6:
    st.header("Detección de anomalías y patrones atípicos")
    
    st.markdown("""Explora la presencia de comportamientos atípicos en las licencias médicas, identificando registros que se alejan 
    del patrón general y que pueden representar situaciones excepcionales o de interés para el análisis.""")

    st.subheader("¿Qué trabajadores reciben licencias repetidas o con altos días acumulados?")

    # Agrupar por RunTrabajador para obtener el total de NumeroDias por cada uno
    dias_por_trabajador = df_licencias_optimizada.groupby('RunTrabajador')['NumeroDias'].sum().reset_index().sort_values(by= 'NumeroDias',ascending = False)

    # Calcular el percentil 95 sobre ese total agrupado
    quantile_95_agrupado = dias_por_trabajador['NumeroDias'].quantile(0.95)
    print(f"El percentil 95 de NumeroDias (total por trabajador) es: {quantile_95_agrupado}")

    ruts_sobre_quantile = dias_por_trabajador[dias_por_trabajador['NumeroDias'] > quantile_95_agrupado]
    

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
    st.plotly_chart(fig, use_container_width=True, key="grafico_anomalias_dias")
    use_container_width=True

    st.info("""**Análisis de Anomalias por Días Acumulados**""")
    st.markdown("""Se expone una distribución de días acumulados altamente asimétrica, donde la mayoría de los cotizantes 
    muestra un comportamiento estándar inferior a los 350 
    días de reposo. En contraste, el uso de la escala logarítmica permite visibilizar con claridad una extensa 
    cola de valores extremos correspondientes al 5% restante de la población; este segmento atípico está compuesto 
    por casos crónicos o de alta severidad que superan los 1.000 días, alcanzando registros aislados de hasta 4.500 
    días acumulados. Identificar este límite resulta crucial para que las Isapres y entidades reguladoras focalicen 
    sus auditorías directamente sobre este grupo minoritario que genera un impacto financiero desproporcionado.
    """)


    st.markdown("---")

    st.subheader("¿Qué profesionales emiten un número anómalo de licencias?")

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

    st.plotly_chart(fig, use_container_width=True, key="grafico_anomalias_profesionales")
    use_container_width=True

    st.info("""**Análisis de Anomalias por Emisión de licencias**""")

    st.markdown(""" Se revela una concentración extrema en la emisión de licencias médicas por parte de un grupo ínfimo de profesionales. 
    Mientras que el límite del Percentil 95 se sitúa en apenas 598 licencias por emisor, lo que significa que la mayoría de médicos se 
    mantiene bajo este umbral estándar, la gráfica de dispersión expone de manera categórica la existencia de "súper emisores" totalmente
     atípicos. Esta evidente desconexión de la norma estándar provee a la Superintendencia de Salud e Isapres una herramienta de fiscalización
     de alta precisión para dirigir auditorías directas hacia conductas de emisión clínicamente improbables o potencialmente fraudulentas.
    """)  

    st.markdown("---")


    st.subheader("¿Existen patrones atípicos por duración o subsidio?")

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
    st.plotly_chart(fig, use_container_width=True, key="grafico_extremos_percentiles")
    use_container_width=True    

    st.info("""**Análisis de Anomalias por Duración y Subsidio**""")

    st.markdown(""" Se expone que la gran mayoría de las licencias se mantiene en rangos bajos de duración y costo, fijando los límites 
    de control del Percentil 95 en apenas 30 días y $1.5M de subsidio por documento. Las licencias individuales que exceden 
    estos umbrales constituyen anomalías que, aunque estadísticamente minoritarias, concentran un impacto financiero y operativo 
    desproporcionado. Aislar valores extremos provee a las Isapres y entidades reguladoras una regla de negocio altamente 
    eficiente para focalizar auditorías directas sobre los casos críticos de mayor costo del sistema.
""")

    st.markdown("---")

    st.subheader("Top de Categorías de Diagnóstico")

    # Agrupar por CategoriaDiagnostico para obtener el total de NumeroDias
    dias_por_categoria = df_licencias_optimizada.groupby('CategoriaDiagnostico')['NumeroDias'].sum().reset_index().sort_values(by = 'NumeroDias', ascending = False)

    # Calcular el percentil 95 sobre el total de días por categoría
    quantile_95_categoria = dias_por_categoria['NumeroDias'].quantile(0.95)
    print(f"El percentil 95 de NumeroDias (total por categoría) es: {quantile_95_categoria}")

    # Identificar y filtrar las categorías que superan el umbral (Top 10)
    resumen_categorias_top = dias_por_categoria[dias_por_categoria['NumeroDias'] > quantile_95_categoria].sort_values(by='NumeroDias', ascending=False).head()

    print("Categorías de diagnóstico sobre el percentil 95:")
    print(resumen_categorias_top)

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
    st.plotly_chart(fig, use_container_width=True, key="grafico_top_categorias_barras")
    use_container_width=True
    # Agrupar correctamente incluyendo cuenta de licencias y suma de días
    dias_por_categoria = df_licencias_optimizada.groupby('CategoriaDiagnostico').agg(
    Dias_Acumulados=('NumeroDias', 'sum')
    ).reset_index()

    # Calcular los Tops
    top_dias_acumuladoss = dias_por_categoria.sort_values(by='Dias_Acumulados', ascending=False).head(10)

    st.info("""**Análisis de Valores Atipicos por Diagnóstico**""")

    st.markdown("""Al aislar las categorías diagnósticas que superan el límite del Percentil 95 de días acumulados, 
    los "Trastornos mentales y del comportamiento" lideran de manera crítica los días autorizados, 
    duplicando con creces el umbral de control técnico. Por su parte, la categoría "Factores que influyen en el estado de salud" se 
    posiciona de forma exacta sobre la línea de referencia. Esta asimetría evidencia que los recursos destinados a licencias de 
    larga duración están impulsados principalmente por patologías psiquiátricas
    """)









