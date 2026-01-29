# 📊 Cómo Abrir el CSV Correctamente

## ✅ PROBLEMA SOLUCIONADO

El archivo `datos_tanques_tilapia.csv` ahora usa **punto y coma (;)** como separador de columnas, lo que es compatible con Excel en configuración regional de español.

## 🔧 Formato del Archivo

- **Separador de columnas:** Punto y coma (`;`)
- **Separador decimal:** Punto (`.`)
- **Codificación:** UTF-8 con BOM

Esto garantiza que:
- Los valores de pH como `7.17` se vean correctamente (no como `717`)
- La temperatura `25.8` se muestre bien (no como `258`)

## 📂 Abrir en Excel

### Método 1: Doble Click (Recomendado)
Simplemente haz doble click en el archivo `datos_tanques_tilapia.csv` y Excel lo abrirá correctamente automáticamente.

### Método 2: Importar Datos
Si el doble click no funciona:

1. Abre Excel
2. Ve a **Datos** → **Obtener datos** → **Desde archivo** → **Desde texto/CSV**
3. Selecciona `datos_tanques_tilapia.csv`
4. En el asistente, verifica:
   - **Delimitador:** Punto y coma
   - **Codificación:** UTF-8
5. Click en **Cargar**

### Método 3: Abrir con Power Query
1. Abre Excel
2. **Datos** → **Nueva consulta** → **Desde archivo** → **Desde CSV**
3. Selecciona el archivo
4. Ajusta configuración si es necesario

## 🐍 Abrir en Python/Pandas

```python
import pandas as pd

# El archivo ahora se carga con punto y coma
df = pd.read_csv('datos_tanques_tilapia.csv', sep=';', decimal='.')

print(df.head())
```

## 🎨 Abrir en Streamlit

La aplicación `app.py` detecta automáticamente el separador:

```python
# La función cargar_datos() ya está configurada para detectar el formato
df = cargar_datos('datos_tanques_tilapia.csv')
```

## 🔄 Si Necesitas Formato con Comas

Si por alguna razón necesitas el formato con comas (`,`):

```python
import pandas as pd

# Leer con punto y coma
df = pd.read_csv('datos_tanques_tilapia.csv', sep=';', decimal='.')

# Guardar con comas
df.to_csv('datos_tanques_tilapia_comas.csv', index=False, sep=',', decimal='.')
```

## 📋 Estructura del Archivo

El CSV tiene las siguientes columnas:

| Columna | Tipo | Descripción | Ejemplo |
|---------|------|-------------|---------|
| `Tanque` | Texto | Identificador del tanque | `Tanque 1` |
| `pH` | Decimal | Nivel de pH del agua | `7.17` |
| `Temperatura_C` | Decimal | Temperatura en °C | `25.8` |
| `Fecha` | Texto | Fecha de medición | `15-01-25` |
| `Hora` | Texto | Hora de medición | `07:00` |
| `Jornada` | Texto | Momento del día | `am` o `pm` |

## ✅ Ejemplo de Datos Correctos

Así es como deberían verse los datos:

```
Tanque;pH;Temperatura_C;Fecha;Hora;Jornada
Tanque 1;7.17;25.8;15-01-25;07:00;am
Tanque 2;7.20;28.3;15-01-25;07:02;am
Tanque 3;7.06;25.6;15-01-25;07:05;am
```

**Nota:** Los valores decimales mantienen el punto (`.`) como separador decimal.

## 🛠️ Regenerar el Archivo

Si necesitas regenerar el archivo CSV:

```bash
python generate_data.py
```

Esto creará un nuevo `datos_tanques_tilapia.csv` con el formato correcto.

## ❓ Problemas Comunes

### Problema: Excel muestra números sin decimales
**Solución:** El archivo ahora está configurado correctamente. Si persiste:
- Cierra Excel completamente
- Abre de nuevo el archivo
- Verifica la configuración regional de Windows (debe estar en español)

### Problema: Los acentos se ven mal
**Solución:** El archivo usa UTF-8 con BOM. Si ves caracteres extraños:
- Abre con Excel 2016 o superior
- O importa usando "Obtener datos" y selecciona codificación UTF-8

### Problema: La aplicación Streamlit no carga el CSV
**Solución:** La app detecta automáticamente el formato. Si falla:
- Verifica que el archivo `datos_tanques_tilapia.csv` existe
- Ejecuta: `python generate_data.py` para regenerarlo

## 📞 Soporte

Si tienes problemas, revisa:
1. La configuración regional de tu sistema está en español
2. Estás usando Excel 2016 o superior
3. El archivo no está abierto en otro programa

---

**¡Los datos ahora se ven perfectos! 🎉**
