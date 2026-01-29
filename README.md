# 🐟 Sistema de Monitoreo de Tilapia en Acuicultura

Sistema completo de análisis y predicción de parámetros de calidad de agua para la cría de tilapia, desarrollado con Streamlit y modelos de Machine Learning.

## 📋 Características

- **Monitoreo en tiempo real** de pH y temperatura en múltiples tanques
- **Clasificación automática** de niveles (Óptimo, Advertencia, Crítico)
- **Modelo predictivo** con regresión lineal
- **Dashboard interactivo** con visualizaciones dinámicas
- **Análisis temporal** de tendencias y patrones
- **Alertas automáticas** para condiciones críticas
- **Exportación de datos** filtrados

## 🚀 Instalación Rápida

### 1. Clonar el repositorio
```bash
git clone <tu-repositorio>
cd tilapia_in_aquaculture
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Generar datos de ejemplo (opcional)
```bash
python generate_data.py
```

### 4. Ejecutar la aplicación
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📊 Estructura del Proyecto

```
tilapia_in_aquaculture/
│
├── app.py                          # Aplicación principal de Streamlit
├── generate_data.py                # Script para generar datos de ejemplo
├── datos_tanques_tilapia.csv       # Datos de monitoreo (6,300+ registros)
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Este archivo
```

## 📈 Parámetros de Calidad de Agua

### Temperatura (°C)
| Nivel | Rango | Significado |
|-------|-------|-------------|
| 🟢 Óptimo | 20.0 - 33.0 °C | Condición ideal para crecimiento |
| 🟡 Advertencia | < 14.0 o > 34.0 °C | Estrés para los peces |
| 🔴 Crítico | < 8.0 o > 42.0 °C | Peligro de mortalidad |

### pH
| Nivel | Rango | Significado |
|-------|-------|-------------|
| 🟢 Óptimo | 6.5 - 9.0 | Condición ideal |
| 🟡 Advertencia | < 6.5 o > 9.0 | Estrés para los peces |
| 🔴 Crítico | < 4.0 o > 10.0 | Peligro de mortalidad |

## 🎯 Funcionalidades Principales

### 1. Dashboard Principal
- Resumen de métricas clave
- Estado actual de todos los tanques
- Distribución de parámetros
- Alertas críticas

### 2. Análisis Temporal
- Evolución de temperatura y pH a lo largo del tiempo
- Comparación entre jornadas (AM vs PM)
- Filtrado por tanque individual o vista general
- Líneas de referencia para rangos óptimos

### 3. Modelo Predictivo
- **Regresión Lineal** para predecir valores futuros
- Variables independientes:
  - Días transcurridos
  - Hora del día
  - Número de tanque
- Métricas de rendimiento (R², MSE)
- Predictor interactivo manual

### 4. Exploración de Datos
- Resumen estadístico completo
- Matriz de correlación entre variables
- Análisis por estado (distribución)
- Heatmaps de temperatura

### 5. Tabla de Datos Completa
- Filtrado avanzado por múltiples criterios
- Ordenamiento y búsqueda
- Exportación a CSV

## 🔧 Uso del Sistema

### Cargar tus propios datos

1. Prepara un archivo CSV con las siguientes columnas:
   ```
   Tanque, pH, Temperatura_C, Fecha, Hora, Jornada
   ```

2. Ejemplo de formato:
   ```csv
   Tanque,pH,Temperatura_C,Fecha,Hora,Jornada
   Tanque 1,7.06,18.2,25-11-25,8:25,am
   Tanque 2,7.05,18.1,25-11-25,8:28,am
   ```

3. Carga el archivo usando el botón "Cargar datos CSV" en la barra lateral

### Hacer predicciones

1. Ve a la pestaña "🤖 Modelo Predictivo"
2. Selecciona la variable a predecir (Temperatura o pH)
3. El modelo se entrenará automáticamente
4. Usa el predictor interactivo para hacer predicciones manuales

## 📦 Dependencias

- **streamlit** 1.31.0 - Framework web interactivo
- **pandas** 2.2.0 - Manipulación de datos
- **numpy** 1.26.3 - Operaciones numéricas
- **plotly** 5.18.0 - Visualizaciones interactivas
- **scikit-learn** 1.4.0 - Machine Learning
- **matplotlib** 3.8.2 - Gráficos estáticos
- **seaborn** 0.13.1 - Visualizaciones estadísticas

## 🌐 Despliegue en Streamlit Cloud

### Opción 1: Desde GitHub

1. Sube tu proyecto a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Inicia sesión con tu cuenta de GitHub
4. Haz clic en "New app"
5. Selecciona tu repositorio y la rama
6. Especifica el archivo principal: `app.py`
7. Haz clic en "Deploy"

### Opción 2: Configuración Manual

1. Asegúrate de que `requirements.txt` esté actualizado
2. Verifica que `app.py` esté en la raíz del proyecto
3. Incluye el archivo `datos_tanques_tilapia.csv` en el repositorio
4. Sigue los pasos de la Opción 1

## 🧪 Generar Nuevos Datos

Para generar datos simulados personalizados:

```bash
python generate_data.py
```

Esto creará un nuevo archivo `datos_tanques_tilapia.csv` con:
- **6,300+ registros** (315 días × 10 tanques × 2 mediciones/día)
- **Variación estacional** realista de temperatura
- **Variación diurna** (AM vs PM)
- **Distribución normal** con parámetros ajustados

## 📝 Notas Técnicas

### Modelo de Regresión Lineal

El modelo utiliza las siguientes features:
```python
X = [días_transcurridos, hora_del_día, número_de_tanque]
y = temperatura_o_pH
```

**Ventajas:**
- Simple e interpretable
- Rápido de entrenar
- Coeficientes indican importancia de cada variable

**Limitaciones:**
- Asume relaciones lineales
- No captura patrones complejos
- Mejor para predicciones a corto plazo

### Mejoras Futuras

- [ ] Implementar modelos más avanzados (Random Forest, XGBoost)
- [ ] Añadir más parámetros de calidad de agua (oxígeno disuelto, amonio, etc.)
- [ ] Sistema de alertas por correo/SMS
- [ ] Integración con sensores IoT en tiempo real
- [ ] Módulo de recomendaciones automáticas
- [ ] Análisis de costo-beneficio

## 🐛 Solución de Problemas

### Error: "No module named 'streamlit'"
```bash
pip install streamlit
```

### Error: "File not found: datos_tanques_tilapia.csv"
```bash
python generate_data.py
```

### La aplicación no carga
- Verifica que todas las dependencias estén instaladas
- Asegúrate de estar en el directorio correcto
- Intenta: `streamlit run app.py --server.port 8502`

## 📞 Soporte

Si encuentras algún problema o tienes sugerencias:
1. Revisa la sección de Solución de Problemas
2. Verifica que todas las dependencias estén instaladas correctamente
3. Asegúrate de tener Python 3.8 o superior

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Siéntete libre de usar, modificar y distribuir.

## 🙏 Créditos

Desarrollado para el monitoreo y análisis de calidad de agua en sistemas de acuicultura de tilapia.

---

**¡Feliz monitoreo! 🐟💧**
