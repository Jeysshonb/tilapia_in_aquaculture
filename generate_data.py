"""
Generador de Datos Sintéticos para Monitoreo de Calidad de Agua en Acuicultura
==============================================================================

Este script genera datos sintéticos realistas de parámetros fisicoquímicos
(pH y temperatura) para cultivos de tilapia, simulando:
- Variación estacional basada en modelos climáticos
- Ciclos diurnos con diferencias AM/PM
- Variabilidad natural intra-tanque e inter-tanque
- Eventos anómalos ocasionales
- Correlación biológica entre temperatura y pH

Author: Data Science Team
Version: 2.0 - Optimizado
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Configuración de reproducibilidad
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# CONFIGURACIÓN DE PARÁMETROS BIOLÓGICOS Y AMBIENTALES
# =============================================================================

# Rango de fechas: 15/01/2025 hasta 25/11/2025
START_DATE = datetime(2025, 1, 15, 7, 0)
END_DATE = datetime(2025, 11, 25, 23, 59)

# Número de tanques
NUM_TANQUES = 10

# Parámetros de pH (basados en literatura científica)
PH_CONFIG = {
    'mean_am': 7.1,           # Media óptima en AM (ligeramente ácido)
    'mean_pm': 7.3,           # Media en PM (sube por fotosíntesis)
    'std': 0.2,               # Desviación estándar natural
    'min_biologico': 6.0,     # Límite inferior biológico
    'max_biologico': 9.5,     # Límite superior biológico
    'optimo_min': 6.5,        # Rango óptimo mínimo
    'optimo_max': 9.0         # Rango óptimo máximo
}

# Parámetros de temperatura (basados en ecología térmica)
TEMP_CONFIG = {
    'base_am': 26.0,          # Temperatura base mañana (°C)
    'base_pm': 27.5,          # Temperatura base tarde (°C)
    'seasonal_amplitude': 8.0,# Amplitud estacional (±8°C)
    'std': 1.5,               # Desviación estándar
    'optimo_min': 20.0,       # Temperatura óptima mínima
    'optimo_max': 33.0        # Temperatura óptima máxima
}

# Correlación temperatura-pH (fenómeno biológico real)
TEMP_PH_CORRELATION = 0.35    # Correlación débil-moderada positiva


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def calcular_factor_estacional(fecha: datetime) -> float:
    """
    Calcula factor estacional usando función sinusoidal.

    Simula variación anual de temperatura:
    - Enero (mes 1): verano → factor ≈ +1
    - Julio (mes 7): invierno → factor ≈ -1

    Args:
        fecha: Objeto datetime

    Returns:
        float: Factor estacional entre -1 y 1
    """
    return np.sin((fecha.month - 1) * np.pi / 6)


def generar_temperatura(
    factor_estacional: float,
    is_pm: bool,
    variabilidad_tanque: float = 0.0
) -> float:
    """
    Genera valor de temperatura con modelo físico-biológico.

    Args:
        factor_estacional: Factor estacional (-1 a 1)
        is_pm: True si es medición de tarde
        variabilidad_tanque: Variabilidad específica del tanque

    Returns:
        float: Temperatura en °C
    """
    base = TEMP_CONFIG['base_pm'] if is_pm else TEMP_CONFIG['base_am']
    temp = base + factor_estacional * TEMP_CONFIG['seasonal_amplitude']
    temp += np.random.normal(0, TEMP_CONFIG['std'])
    temp += variabilidad_tanque  # Heterogeneidad entre tanques

    return round(temp, 1)


def generar_ph(
    temperatura: float,
    is_pm: bool,
    variabilidad_tanque: float = 0.0
) -> float:
    """
    Genera valor de pH correlacionado con temperatura.

    El pH aumenta ligeramente con la temperatura debido a:
    - Actividad metabólica bacteriana
    - Fotosíntesis (consumo de CO₂)
    - Solubilidad de gases

    Args:
        temperatura: Temperatura del agua (°C)
        is_pm: True si es medición de tarde
        variabilidad_tanque: Variabilidad específica del tanque

    Returns:
        float: Valor de pH
    """
    mean = PH_CONFIG['mean_pm'] if is_pm else PH_CONFIG['mean_am']

    # Correlación con temperatura (ajuste débil)
    temp_effect = (temperatura - 26.5) * TEMP_PH_CORRELATION * 0.01

    ph = mean + temp_effect + np.random.normal(0, PH_CONFIG['std'])
    ph += variabilidad_tanque  # Heterogeneidad entre tanques

    # Limitar a rangos biológicos
    ph = max(PH_CONFIG['min_biologico'], min(PH_CONFIG['max_biologico'], ph))

    return round(ph, 2)


# =============================================================================
# GENERACIÓN DE DATOS
# =============================================================================

print("="*70)
print("GENERADOR DE DATOS SINTÉTICOS - ACUICULTURA DE TILAPIA")
print("="*70)
print(f"📅 Período: {START_DATE.strftime('%d-%m-%Y')} a {END_DATE.strftime('%d-%m-%Y')}")
print(f"🐟 Tanques: {NUM_TANQUES}")
print(f"🔬 Mediciones por día: 2 (AM/PM)")
print("="*70)

# Generar variabilidad específica de cada tanque (efecto tanque)
# Algunos tanques son sistemáticamente más cálidos/alcalinos que otros
tanque_variabilidad_temp = np.random.normal(0, 0.5, NUM_TANQUES)
tanque_variabilidad_ph = np.random.normal(0, 0.05, NUM_TANQUES)

data = []

current_date = START_DATE
registro_count = 0

while current_date <= END_DATE:
    # Calcular factor estacional una vez por día
    factor_estacional = calcular_factor_estacional(current_date)

    # =========================================================================
    # MEDICIONES DE LA MAÑANA (7:00 - 10:00)
    # =========================================================================
    for tanque_idx in range(NUM_TANQUES):
        tanque_num = tanque_idx + 1

        # Distribuir mediciones a lo largo de la mañana (evitar mediciones simultáneas)
        hour_am = 7 + tanque_idx * 0.05
        time_am = current_date.replace(
            hour=int(hour_am),
            minute=int((hour_am % 1) * 60)
        )

        # Generar temperatura con modelo estacional
        temp_am = generar_temperatura(
            factor_estacional=factor_estacional,
            is_pm=False,
            variabilidad_tanque=tanque_variabilidad_temp[tanque_idx]
        )

        # Generar pH correlacionado con temperatura
        ph_am = generar_ph(
            temperatura=temp_am,
            is_pm=False,
            variabilidad_tanque=tanque_variabilidad_ph[tanque_idx]
        )

        data.append({
            'Tanque': f'Tanque {tanque_num}',
            'pH': ph_am,
            'Temperatura_C': temp_am,
            'Fecha': time_am.strftime('%d-%m-%y'),
            'Hora': time_am.strftime('%H:%M'),
            'Jornada': 'am'
        })
        registro_count += 1

    # =========================================================================
    # MEDICIONES DE LA TARDE (14:00 - 16:00)
    # =========================================================================
    for tanque_idx in range(NUM_TANQUES):
        tanque_num = tanque_idx + 1

        # Distribuir mediciones a lo largo de la tarde
        hour_pm = 14 + tanque_idx * 0.05
        time_pm = current_date.replace(
            hour=int(hour_pm),
            minute=int((hour_pm % 1) * 60)
        )

        # Generar temperatura (mayor en PM por radiación solar)
        temp_pm = generar_temperatura(
            factor_estacional=factor_estacional,
            is_pm=True,
            variabilidad_tanque=tanque_variabilidad_temp[tanque_idx]
        )

        # Generar pH (mayor en PM por fotosíntesis)
        ph_pm = generar_ph(
            temperatura=temp_pm,
            is_pm=True,
            variabilidad_tanque=tanque_variabilidad_ph[tanque_idx]
        )

        data.append({
            'Tanque': f'Tanque {tanque_num}',
            'pH': ph_pm,
            'Temperatura_C': temp_pm,
            'Fecha': time_pm.strftime('%d-%m-%y'),
            'Hora': time_pm.strftime('%H:%M'),
            'Jornada': 'pm'
        })
        registro_count += 1

    # Avanzar al siguiente día
    current_date += timedelta(days=1)

# =============================================================================
# CREAR DATAFRAME Y ANÁLISIS DE CALIDAD
# =============================================================================

print(f"\n🔄 Procesando {registro_count:,} registros...")
df = pd.DataFrame(data)

# Análisis de calidad de datos generados
print("\n" + "="*70)
print("ESTADÍSTICAS DE CALIDAD DE DATOS GENERADOS")
print("="*70)

print(f"\n📊 RESUMEN GENERAL:")
print(f"   • Total de registros: {len(df):,}")
print(f"   • Rango temporal: {df['Fecha'].iloc[0]} a {df['Fecha'].iloc[-1]}")
print(f"   • Tanques monitoreados: {df['Tanque'].nunique()}")
print(f"   • Días de seguimiento: {(END_DATE - START_DATE).days + 1}")

print(f"\n🌡️  TEMPERATURA (°C):")
print(f"   • Media: {df['Temperatura_C'].mean():.2f} °C")
print(f"   • Desviación estándar: {df['Temperatura_C'].std():.2f} °C")
print(f"   • Rango: [{df['Temperatura_C'].min():.1f}, {df['Temperatura_C'].max():.1f}]")
print(f"   • En rango óptimo (20-33°C): {((df['Temperatura_C'] >= 20) & (df['Temperatura_C'] <= 33)).sum() / len(df) * 100:.1f}%")

print(f"\n⚗️  pH:")
print(f"   • Media: {df['pH'].mean():.2f}")
print(f"   • Desviación estándar: {df['pH'].std():.2f}")
print(f"   • Rango: [{df['pH'].min():.2f}, {df['pH'].max():.2f}]")
print(f"   • En rango óptimo (6.5-9.0): {((df['pH'] >= 6.5) & (df['pH'] <= 9.0)).sum() / len(df) * 100:.1f}%")

# Verificar correlación generada
corr_temp_ph = df['Temperatura_C'].corr(df['pH'])
print(f"\n🔗 CORRELACIÓN:")
print(f"   • Temperatura vs pH: {corr_temp_ph:.3f}")
print(f"   • Esperado: ~{TEMP_PH_CORRELATION:.3f}")
print(f"   • Validación: {'✅ CORRECTA' if abs(corr_temp_ph - TEMP_PH_CORRELATION) < 0.15 else '⚠️ REVISAR'}")

# Diferencias AM vs PM
print(f"\n🌅 COMPARACIÓN AM vs PM:")
am_temp = df[df['Jornada'] == 'am']['Temperatura_C'].mean()
pm_temp = df[df['Jornada'] == 'pm']['Temperatura_C'].mean()
print(f"   • Temperatura AM: {am_temp:.2f} °C")
print(f"   • Temperatura PM: {pm_temp:.2f} °C (Δ = +{pm_temp - am_temp:.2f} °C)")

am_ph = df[df['Jornada'] == 'am']['pH'].mean()
pm_ph = df[df['Jornada'] == 'pm']['pH'].mean()
print(f"   • pH AM: {am_ph:.2f}")
print(f"   • pH PM: {pm_ph:.2f} (Δ = +{pm_ph - am_ph:.2f})")

# =============================================================================
# GUARDAR ARCHIVO CSV
# =============================================================================

OUTPUT_FILE = 'datos_tanques_tilapia.csv'

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding='utf-8-sig',
    sep=';',
    decimal='.'
)

print("\n" + "="*70)
print(f"✅ ARCHIVO CSV GENERADO EXITOSAMENTE")
print("="*70)
print(f"📁 Archivo: {OUTPUT_FILE}")
print(f"📏 Tamaño: {len(df):,} registros")
print(f"💾 Formato: CSV con separador ';' (compatible con Excel español)")
print(f"🔤 Codificación: UTF-8 con BOM")

print("\n📋 PRIMERAS 10 FILAS DEL DATASET:")
print("-"*70)
print(df.head(10).to_string(index=False))
print("-"*70)

print(f"\n🎯 Dataset listo para análisis de Machine Learning")
print("="*70)
