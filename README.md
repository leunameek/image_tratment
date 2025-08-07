# Image Processing Laboratory 1 - Mathematical Documentation

## English Version

### Overview
This project implements various image processing algorithms including geometric transformations, contrast adjustments, and normalization techniques. All algorithms are implemented from scratch without using built-in functions for educational purposes.

### Mathematical Concepts

#### 1. Geometric Transformations

##### 1.1 Image Scaling (Downscaling)
**Algorithm**: Custom bilinear interpolation downscaling
**Mathematical Formula**:
- New dimensions: `new_h = h // k`, `new_w = w // k`
- Source mapping: `src_i = (i + 0.5) * k`, `src_j = (j + 0.5) * k`
- Bilinear interpolation:
  ```
  result[i,j] = (1-wi)*(1-wj)*p11 + (1-wi)*wj*p12 + wi*(1-wj)*p21 + wi*wj*p22
  ```
  where `wi = src_i - i1`, `wj = src_j - j1`

##### 1.2 Image Scaling (Upscaling)
**Algorithm**: Custom bilinear interpolation upscaling
**Mathematical Formula**:
- New dimensions: `new_h = h * k`, `new_w = w * k`
- Source mapping: `src_i = i / k`, `src_j = j / k`
- Same bilinear interpolation formula as downscaling

##### 1.3 Image Rotation
**Algorithm**: Affine transformation with rotation matrix
**Mathematical Formula**:
```
Rotation Matrix M = [cos(θ)  -sin(θ)  tx]
                    [sin(θ)   cos(θ)  ty]
                    [  0       0       1 ]

where:
- θ = angle in radians
- tx, ty = translation components
- New dimensions calculated to accommodate rotated image
```

#### 2. Contrast and Brightness Adjustments

##### 2.1 Image Negative
**Formula**: `output = 255 - input`
**Mathematical Basis**: Inverts the intensity values across the 0-255 range

##### 2.2 Brightness Adjustment
**Formula**: `output = clip(input ± delta, 0, 255)`
**Mathematical Basis**: Linear shift of intensity values by a constant delta

##### 2.3 Gamma Correction
**Formula**: `output = 255 * (input/255)^(1/γ)`
**Mathematical Basis**: Power law transformation to correct for display gamma
- γ > 1: Darkens the image
- γ < 1: Brightens the image
- γ = 1: No change

##### 2.4 Logarithmic Contrast
**Formula**: `output = 255 * log(1 + input) / log(256)`
**Mathematical Basis**: Logarithmic transformation to enhance dark regions
- Compresses bright values and expands dark values
- Uses log1p for numerical stability

##### 2.5 Exponential Contrast
**Formula**: `output = 255 * (exp(α * input/255) - 1) / (exp(α) - 1)`
**Mathematical Basis**: Exponential transformation to enhance bright regions
- α > 0: Controls the degree of enhancement
- Uses expm1 for numerical stability

#### 3. Image Normalization

##### 3.1 Normalization to [0, 100]
**Formula**: `output = (input - min) * 100 / (max - min)`
**Mathematical Basis**: Linear scaling to range [0, 100]
- Preserves relative intensity relationships
- Handles edge case when max = min

##### 3.2 Normalization to [-1, 1]
**Formula**: `output = (input - min) * 2 / (max - min) - 1`
**Mathematical Basis**: Linear scaling to range [-1, 1]
- Centers the data around zero
- Useful for machine learning applications

#### 4. Histogram Computation
**Algorithm**: Custom histogram without built-in functions
**Mathematical Formula**:
```
counts[i] = sum(input == i) for i in [0, 255]
```
**Implementation**: Uses `np.bincount()` for efficient counting

#### 5. Dynamic Text Scaling
**Algorithm**: Adaptive text and banner sizing
**Mathematical Formula**:
- Banner height: `banner_h = max(30, min(80, width * 0.05))`
- Font scale: `font_scale = max(0.3, min(1.5, width / 1000.0))`
- Text positioning: Centered using `cv2.getTextSize()`

### Algorithm Complexity
- **Downscaling/Upscaling**: O(new_h × new_w)
- **Rotation**: O(new_h × new_w)
- **Contrast adjustments**: O(h × w)
- **Normalization**: O(h × w)
- **Histogram**: O(h × w)

---

## Versión en Español

### Descripción General
Este proyecto implementa varios algoritmos de procesamiento de imágenes incluyendo transformaciones geométricas, ajustes de contraste y técnicas de normalización. Todos los algoritmos están implementados desde cero sin usar funciones integradas con fines educativos.

### Conceptos Matemáticos

#### 1. Transformaciones Geométricas

##### 1.1 Escalado de Imagen (Reducción)
**Algoritmo**: Reducción personalizada con interpolación bilineal
**Fórmula Matemática**:
- Nuevas dimensiones: `new_h = h // k`, `new_w = w // k`
- Mapeo de origen: `src_i = (i + 0.5) * k`, `src_j = (j + 0.5) * k`
- Interpolación bilineal:
  ```
  result[i,j] = (1-wi)*(1-wj)*p11 + (1-wi)*wj*p12 + wi*(1-wj)*p21 + wi*wj*p22
  ```
  donde `wi = src_i - i1`, `wj = src_j - j1`

##### 1.2 Escalado de Imagen (Aumento)
**Algoritmo**: Aumento personalizado con interpolación bilineal
**Fórmula Matemática**:
- Nuevas dimensiones: `new_h = h * k`, `new_w = w * k`
- Mapeo de origen: `src_i = i / k`, `src_j = j / k`
- Misma fórmula de interpolación bilineal que la reducción

##### 1.3 Rotación de Imagen
**Algoritmo**: Transformación afín con matriz de rotación
**Fórmula Matemática**:
```
Matriz de Rotación M = [cos(θ)  -sin(θ)  tx]
                       [sin(θ)   cos(θ)  ty]
                       [  0       0       1 ]

donde:
- θ = ángulo en radianes
- tx, ty = componentes de traslación
- Nuevas dimensiones calculadas para acomodar la imagen rotada
```

#### 2. Ajustes de Contraste y Brillo

##### 2.1 Negativo de Imagen
**Fórmula**: `output = 255 - input`
**Base Matemática**: Invierte los valores de intensidad en el rango 0-255

##### 2.2 Ajuste de Brillo
**Fórmula**: `output = clip(input ± delta, 0, 255)`
**Base Matemática**: Desplazamiento lineal de valores de intensidad por una constante delta

##### 2.3 Corrección Gamma
**Fórmula**: `output = 255 * (input/255)^(1/γ)`
**Base Matemática**: Transformación de ley de potencia para corregir gamma de pantalla
- γ > 1: Oscurece la imagen
- γ < 1: Aclara la imagen
- γ = 1: Sin cambios

##### 2.4 Contraste Logarítmico
**Fórmula**: `output = 255 * log(1 + input) / log(256)`
**Base Matemática**: Transformación logarítmica para realzar regiones oscuras
- Comprime valores brillantes y expande valores oscuros
- Usa log1p para estabilidad numérica

##### 2.5 Contraste Exponencial
**Fórmula**: `output = 255 * (exp(α * input/255) - 1) / (exp(α) - 1)`
**Base Matemática**: Transformación exponencial para realzar regiones brillantes
- α > 0: Controla el grado de realce
- Usa expm1 para estabilidad numérica

#### 3. Normalización de Imagen

##### 3.1 Normalización a [0, 100]
**Fórmula**: `output = (input - min) * 100 / (max - min)`
**Base Matemática**: Escalado lineal al rango [0, 100]
- Preserva relaciones relativas de intensidad
- Maneja caso límite cuando max = min

##### 3.2 Normalización a [-1, 1]
**Fórmula**: `output = (input - min) * 2 / (max - min) - 1`
**Base Matemática**: Escalado lineal al rango [-1, 1]
- Centra los datos alrededor de cero
- Útil para aplicaciones de machine learning

#### 4. Cálculo de Histograma
**Algoritmo**: Histograma personalizado sin funciones integradas
**Fórmula Matemática**:
```
counts[i] = sum(input == i) para i en [0, 255]
```
**Implementación**: Usa `np.bincount()` para conteo eficiente

#### 5. Escalado Dinámico de Texto
**Algoritmo**: Tamaño adaptativo de texto y banner
**Fórmula Matemática**:
- Altura del banner: `banner_h = max(30, min(80, width * 0.05))`
- Escala de fuente: `font_scale = max(0.3, min(1.5, width / 1000.0))`
- Posicionamiento de texto: Centrado usando `cv2.getTextSize()`

### Complejidad de Algoritmos
- **Reducción/Aumento**: O(new_h × new_w)
- **Rotación**: O(new_h × new_w)
- **Ajustes de contraste**: O(h × w)
- **Normalización**: O(h × w)
- **Histograma**: O(h × w)

### Uso del Programa

```bash
# Ejecutar con imagen de entrada
python corrected.py --imagen image.png --salida ./outputs

# Parámetros opcionales
python corrected.py --imagen image.png --gamma 2.2 --brillo 40 --alpha_exp 2.0
```

### Archivos de Salida
- Imágenes procesadas con títulos y dimensiones
- Histogramas correspondientes
- Archivos de normalización (.npy) para análisis posterior

### Dependencias
- Python 3.8+
- OpenCV (cv2)
- NumPy
- Matplotlib
- Pathlib 