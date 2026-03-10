# 🎵 Lab 2.1: Compositor de Música y Letras

## 🎯 **Objetivo del Laboratorio**

Desarrollar un **sistema de composición musical** que pueda generar **música y letras de canciones** en **diferentes géneros musicales** utilizando **Redes Neuronales Recurrentes (RNN/LSTM)** y **modelos Transformers**.

---

## 📋 **Contenido del Laboratorio**

### **🎯 Objetivos de Aprendizaje:**
- **Procesamiento de Audio**: MFCC, espectrogramas, análisis de audio
- **Modelos Secuenciales**: RNN, LSTM, GRU para generación musical
- **NLP Creativo**: Generación de letras, patrones líricos
- **Modelos Transformers**: Aplicaciones en generación de texto y música
- **Géneros Musicales**: Características y estructuras por género

### **🔧 Tecnologías Utilizadas:**
- **TensorFlow/Keras**: Para modelos RNN/LSTM
- **Librosa**: Para procesamiento de audio
- **Magenta**: Herramientas de Google para música y arte
- **Music21**: Para análisis musical teórico
- **Transformers**: Para generación avanzada de texto

---

## 🏗️ **Estructura del Laboratorio**

```
Lab-2.1-Compositor-Musica/
├── 📁 src/                           # Código fuente
│   ├── 📄 music_generator.py          # Generador musical principal
│   ├── 📄 lyrics_generator.py        # Generador de letras
│   ├── 📄 genre_classifier.py       # Clasificador de géneros
│   ├── 📄 audio_processor.py        # Procesamiento de audio
│   ├── 📄 midi_converter.py         # Conversión MIDI
│   └── 📄 composer_interface.py     # Interfaz del compositor
│
├── 📁 models/                        # Modelos entrenados
│   ├── 📁 music_models/              # Modelos de música
│   │   ├── 📄 rock_model.h5         # Modelo de rock
│   │   ├── 📄 jazz_model.h5         # Modelo de jazz
│   │   ├── 📄 classical_model.h5     # Modelo clásica
│   │   └── 📄 hiphop_model.h5        # Modelo hip-hop
│   ├── 📁 lyrics_models/            # Modelos de letras
│   │   ├── 📄 pop_lyrics_model.h5    # Modelo pop
│   │   ├── 📄 rock_lyrics_model.h5   # Modelo rock
│   │   └── 📄 rap_lyrics_model.h5    # Modelo rap
│   └── 📁 genre_models/             # Modelos de clasificación
│
├── 📁 data/                          # Dataset y datos
│   ├── 📁 midi_files/               # Archivos MIDI por género
│   │   ├── 📁 rock/                # MIDI de rock
│   │   ├── 📁 jazz/                # MIDI de jazz
│   │   ├── 📁 classical/            # MIDI clásica
│   │   ├── 📁 pop/                 # Pop
│   │   └── 📁 hiphop/              # Hip-hop
│   ├── 📁 lyrics/                  # Letras por género
│   │   ├── 📁 rock_lyrics/          # Letras de rock
│   │   ├── 📁 pop_lyrics/           # Letras de pop
│   │   └── 📁 rap_lyrics/           # Letras de rap
│   └── 📁 processed/               # Datos preprocesados
│
├── 📁 outputs/                       # Resultados generados
│   ├── 📁 generated_music/          # Música generada
│   ├── 📁 generated_lyrics/        # Letras generadas
│   ├── 📁 complete_songs/          # Canciones completas
│   └── 📁 midi_files/              # Archivos MIDI exportados
│
├── 📁 notebooks/                     # Jupyter notebooks
│   ├── 📄 01_data_exploration.ipynb # Exploración de datos
│   ├── 📄 02_music_model_training.ipynb # Entrenamiento música
│   ├── 📄 03_lyrics_model_training.ipynb # Entrenamiento letras
│   └── 📄 04_generation_demo.ipynb # Demostración de generación
│
└── 📄 requirements.txt              # Dependencias específicas
```

---

## 🎵 **Géneros Musicales Implementados**

### **🎸 Rock/Pop:**
- **Características**: 4/4, power chords, estructura verso-estribillo
- **Instrumentación**: Guitarra eléctrica, bajo, batería, voz
- **Patrones Líricos**: Historias de amor, rebelión, vida cotidiana
- **Tempo**: 120-140 BPM

### **🎹 Jazz/Blues:**
- **Características**: Swing, improvisación, acordes extendidos
- **Instrumentación**: Piano, contrabajo, batería, saxofón
- **Patrones Líricos**: Melancolía, expresión emocional, narrativa
- **Tempo**: 60-120 BPM

### **🎻 Clásica:**
- **Características**: Estructura compleja, orquestación, formas clásicas
- **Instrumentación**: Orquesta completa, piano, cuerdas
- **Patrones Líricos**: Poesía clásica, temas universales
- **Tempo**: Variable según forma musical

### **🎤 Hip-Hop/Rap:**
- **Características**: Ritmo fuerte, rimas, spoken word
- **Instrumentación**: Beat box, samples, sintetizadores
- **Patrones Líricos**: Rimas AABB, storytelling, urban life
- **Tempo**: 80-100 BPM

### **🎹 Electronic/EDM:**
- **Características**: Síntesis, loops, drops, efectos digitales
- **Instrumentación**: Sintetizadores, drum machines, samplers
- **Patrones Líricos**: Futurismo, tecnología, energía
- **Tempo**: 128-140 BPM

---

## 🚀 **Flujo de Trabajo**

### **📋 Paso 1: Preparación del Entorno**
```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### **📋 Paso 2: Exploración de Datos**
```bash
# Ejecutar notebook de exploración
jupyter notebook notebooks/01_data_exploration.ipynb
```

### **📋 Paso 3: Entrenamiento de Modelos**
```bash
# Entrenar modelo de música
python src/music_generator.py --genre rock --train

# Entrenar modelo de letras
python src/lyrics_generator.py --genre pop --train

# O usar notebooks interactivos
jupyter notebook notebooks/02_music_model_training.ipynb
jupyter notebook notebooks/03_lyrics_model_training.ipynb
```

### **📋 Paso 4: Generación de Música y Letras**
```bash
# Generar música
python src/music_generator.py --genre jazz --length 120 --output jazz_song.mid

# Generar letras
python src/lyrics_generator.py --genre rap --length 32 --output rap_lyrics.txt

# Generar canción completa
python src/composer_interface.py --genre rock --complete
```

---

## 🧠 **Arquitectura del Modelo Musical**

### **🏗️ Estructura RNN/LSTM:**
```
Input (sequence_length, feature_dim)
    ↓
LSTM(256, return_sequences=True)
    ↓
Dropout(0.3)
    ↓
LSTM(256, return_sequences=True)
    ↓
Dropout(0.3)
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.2)
    ↓
Dense(output_dim, activation='softmax')
```

### **📊 Parámetros del Modelo:**
- **Input**: Secuencias de 32 pasos temporales
- **Features**: 128 dimensiones (pitch, velocity, duration)
- **Capas LSTM**: 2 capas con 256 unidades cada una
- **Regularización**: Dropout 0.3 entre capas
- **Output**: Distribución sobre notas MIDI (0-127)

---

## 💬 **Arquitectura del Modelo de Letras**

### **🏗️ Estructura Transformer:**
```
Input (sequence_length)
    ↓
Embedding(vocab_size, 256)
    ↓
Positional Encoding
    ↓
Multi-Head Attention (8 heads, 64 dim)
    ↓
Feed-Forward Network (512, 2048, 512)
    ↓
Layer Normalization
    ↓
Output Dense (vocab_size, activation='softmax')
```

### **📊 Parámetros del Modelo:**
- **Vocab Size**: 10,000 tokens
- **Sequence Length**: 128 tokens
- **Embedding Dim**: 256
- **Attention Heads**: 8
- **Feed Forward**: 512 → 2048 → 512

---

## 🎯 **Uso del Sistema**

### **🎸 Generación Musical:**
```python
from src.music_generator import MusicGenerator

# Inicializar generador
generator = MusicGenerator(model_path='models/rock_model.h5')

# Generar secuencia musical
music_sequence = generator.generate(
    genre='rock',
    length=120,  # segundos
    temperature=0.8,
    seed_melody=[60, 64, 67]  # Do, Mi, Sol
)

# Guardar como MIDI
generator.save_midi(music_sequence, 'rock_song.mid')
```

### **🎤 Generación de Letras:**
```python
from src.lyrics_generator import LyricsGenerator

# Inicializar generador
lyrics_gen = LyricsGenerator(model_path='models/rap_lyrics_model.h5')

# Generar letras
lyrics = lyrics_gen.generate(
    genre='rap',
    length=32,  # líneas
    temperature=0.7,
    theme='city_life'
)

print(lyrics)
```

### **🎵 Canción Completa:**
```python
from src.composer_interface import ComposerInterface

# Inicializar compositor
composer = ComposerInterface()

# Generar canción completa
song = composer.create_song(
    genre='pop',
    structure=['intro', 'verse', 'chorus', 'verse', 'chorus', 'bridge', 'chorus', 'outro'],
    theme='love_story',
    duration=180  # segundos
)

# Exportar en múltiples formatos
composer.export_song(song, 'pop_love_song', formats=['midi', 'mp3', 'pdf'])
```

---

## 🔧 **Configuración y Parámetros**

### **📄 Configuración Musical:**
```python
MUSIC_CONFIG = {
    'sequence_length': 32,
    'feature_dim': 128,
    'midi_range': (21, 108),  # Piano keys
    'tempo_ranges': {
        'rock': (120, 140),
        'jazz': (60, 120),
        'classical': (40, 180),
        'hiphop': (80, 100),
        'electronic': (128, 140)
    }
}
```

### **📄 Configuración de Letras:**
```python
LYRICS_CONFIG = {
    'vocab_size': 10000,
    'max_sequence_length': 128,
    'min_line_length': 4,
    'max_line_length': 16,
    'rhyme_schemes': {
        'pop': ['AABB', 'ABAB'],
        'rock': ['AABB', 'ABCB'],
        'rap': ['AABB', 'AAAA', 'ABAB'],
        'jazz': ['AABA', 'ABAC']
    }
}
```

---

## 📊 **Métricas de Evaluación**

### **🎵 Métricas Musicales:**
- **Tonal Consistency**: Coherencia tonal de la melodía
- **Rhythmic Variety**: Diversidad rítmica
- **Harmonic Correctness**: Corrección armónica
- **Genre Accuracy**: Similitud con género objetivo
- **Musicality**: Calidad musical percibida

### **💬 Métricas de Letras:**
- **Coherence**: Coherencia temática
- **Rhyme Quality**: Calidad de las rimas
- **Vocabulary Richness**: Riqueza léxica
- **Genre Appropriateness**: Adecuación al género
- **Emotional Impact**: Impacto emocional

---

## 🚨 **Desafíos Comunes y Soluciones**

### **❌ Problema: Música Incoherente**
- **Síntomas**: Melodía sin sentido, saltos aleatorios
- **Solución**: Aumentar temperatura, mejorar preprocesamiento, usar attention

### **❌ Problema: Letras Repetitivas**
- **Síntomas**: Mismas frases, vocabulario limitado
- **Solución**: Aumentar vocabulario, usar sampling top-k/top-p

### **❌ Problema: Fuga de Género**
- **Síntomas**: Rock suena como clásica, rap suena como pop
- **Solución**: Mejor clasificación de géneros, fine-tuning específico

---

## 🎮 **Interfaz del Compositor**

### **🖥️ Modo Interactivo:**
```python
# Iniciar interfaz interactiva
python src/composer_interface.py --interactive

# Menú de opciones:
# 1. Generar música
# 2. Generar letras
# 3. Crear canción completa
# 4. Combinar géneros
# 5. Exportar proyecto
```

### **🎨 Modo GUI:**
```python
# Iniciar interfaz gráfica
python src/composer_interface.py --gui

# Características:
# - Visualización de notas
# - Reproducción en tiempo real
# - Edición de parámetros
# - Exportación múltiple formatos
```

---

## 📈 **Extensiones y Mejoras**

### **🚀 Mejoras Futuras:**
1. **Multi-Genre**: Combinación de múltiples géneros
2. **Voice Synthesis**: Síntesis de voz para las letras
3. **Real-time Generation**: Generación en tiempo real
4. **Collaborative AI**: Colaboración con humanos
5. **Music Theory Integration**: Integración de teoría musical

### **🌐 Aplicaciones del Mundo Real:**
- **Production Studios**: Asistencia en composición
- **Music Education**: Herramienta de aprendizaje
- **Content Creation**: Música para videos y streams
- **Therapy**: Musicoterapia personalizada

---

## 📚 **Recursos Adicionales**

### **📖 Documentación:**
- [Magenta Documentation](https://magenta.tensorflow.org/)
- [Music21 Documentation](https://web.mit.edu/music21/)
- [Librosa Documentation](https://librosa.org/doc/main/)

### **📊 Datasets Recomendados:**
- **Lakh MIDI Dataset**: 176,581 archivos MIDI
- **Million Song Dataset**: Metadatos de 1M canciones
- **MAESTRO**: Dataset de piano de Google
- **Genius Lyrics**: Letras con anotaciones

---

## 🎯 **Evaluación del Laboratorio**

### **📋 Criterios de Éxito:**
- [ ] **Modelos Entrenados**: Mínimo 3 géneros funcionales
- [ ] **Generación Musical**: Música coherente y con género
- [ ] **Generación de Letras**: Letras coherentes y temáticas
- [ ] **Interfaz Funcional**: Sistema completo de composición
- [ ] **Exportación**: Múltiples formatos de salida

### **🏆 Niveles de Logro:**
- **🥉 Básico**: 2 géneros con generación funcional
- **🥈 Intermedio**: 3+ géneros con interfaz completa
- **🥇 Avanzado**: Sistema completo con innovaciones

---

## 🎉 **Conclusión**

Este laboratorio te permitirá crear un **sistema completo de composición musical** que combina **técnicas avanzadas de IA** con **expresión artística**, abriendo nuevas posibilidades en la creación musical asistida por inteligencia.

**🚀 ¡Conviértete en un compositor del futuro con IA!**

*La música del mañana está siendo compuesta hoy.*
