import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuración
np.random.seed(42)

# Rango de fechas: 15/01/2025 hasta 25/11/2025
start_date = datetime(2025, 1, 15, 7, 0)
end_date = datetime(2025, 11, 25, 23, 59)

# Generar datos
data = []

current_date = start_date
while current_date <= end_date:
    # Medición de la mañana (7:00 - 10:00)
    for tanque in range(1, 11):
        hour_am = 7 + (tanque - 1) * 0.05  # Distribuir las mediciones
        time_am = current_date.replace(
            hour=int(hour_am),
            minute=int((hour_am % 1) * 60)
        )

        # pH: rango óptimo 6.5-9.0, con variación natural
        ph_am = np.random.normal(7.1, 0.15)  # Media 7.1, desviación 0.15
        ph_am = round(max(6.0, min(9.5, ph_am)), 2)

        # Temperatura: rango óptimo 20-33°C
        # Variación por época del año (enero = verano, julio = invierno en hemisferio sur)
        month_factor = np.sin((current_date.month - 1) * np.pi / 6)  # Ciclo anual
        temp_base = 26 + month_factor * 8  # Oscila entre 18°C y 34°C
        temp_am = temp_base + np.random.normal(0, 1.5)
        temp_am = round(temp_am, 1)

        data.append({
            'Tanque': f'Tanque {tanque}',
            'pH': ph_am,
            'Temperatura_C': temp_am,
            'Fecha': time_am.strftime('%d-%m-%y'),
            'Hora': time_am.strftime('%H:%M'),
            'Jornada': 'am'
        })

    # Medición de la tarde (14:00 - 16:00)
    for tanque in range(1, 11):
        hour_pm = 14 + (tanque - 1) * 0.05
        time_pm = current_date.replace(
            hour=int(hour_pm),
            minute=int((hour_pm % 1) * 60)
        )

        # pH tarde (ligeramente más bajo por fotosíntesis)
        ph_pm = np.random.normal(7.2, 0.2)
        ph_pm = round(max(6.0, min(9.5, ph_pm)), 2)

        # Temperatura tarde (más alta que en la mañana)
        month_factor = np.sin((current_date.month - 1) * np.pi / 6)
        temp_base = 27 + month_factor * 8  # Un poco más alta
        temp_pm = temp_base + np.random.normal(0, 1.5)
        temp_pm = round(temp_pm, 1)

        data.append({
            'Tanque': f'Tanque {tanque}',
            'pH': ph_pm,
            'Temperatura_C': temp_pm,
            'Fecha': time_pm.strftime('%d-%m-%y'),
            'Hora': time_pm.strftime('%H:%M'),
            'Jornada': 'pm'
        })

    # Avanzar al siguiente día
    current_date += timedelta(days=1)

# Crear DataFrame
df = pd.DataFrame(data)

# Guardar CSV con punto y coma (;) para Excel en español
# Esto preserva los puntos decimales correctamente
df.to_csv('datos_tanques_tilapia.csv', index=False, encoding='utf-8-sig', sep=';', decimal='.')

print(f"✅ CSV generado exitosamente!")
print(f"📊 Total de registros: {len(df)}")
print(f"📅 Rango: {df['Fecha'].iloc[0]} hasta {df['Fecha'].iloc[-1]}")
print(f"💡 Formato: Excel en español (separador: punto y coma)")
print(f"\nPrimeras filas:")
print(df.head(10))
