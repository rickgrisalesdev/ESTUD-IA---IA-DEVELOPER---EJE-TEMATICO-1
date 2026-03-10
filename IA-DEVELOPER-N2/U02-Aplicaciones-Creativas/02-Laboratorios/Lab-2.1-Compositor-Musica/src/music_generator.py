#!/usr/bin/env python3
"""
Lab 2.1 - Compositor de Música y Letras
Generador musical usando RNN/LSTM para diferentes géneros

Aplicando buenas prácticas: PEP 8, Type Hinting, SOLID, Docstrings
"""

import numpy as np
import tensorflow as tf
import librosa
import midiutil
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MusicGenre:
    """
    Clase para definir características de géneros musicales.
    
    Aplica el Principio de Responsabilidad Única: Solo se encarga de
    definir las características específicas de cada género musical.
    """
    
    def __init__(self, name: str, tempo_range: Tuple[int, int], 
                 key_signatures: List[str], time_signatures: List[str],
                 chord_progressions: List[List[str]]) -> None:
        """
        Inicializar características del género musical.
        
        Args:
            name (str): Nombre del género.
            tempo_range (Tuple[int, int]): Rango de tempo (BPM).
            key_signatures (List[str]): Tonalidades comunes.
            time_signatures (List[str]): Compases comunes.
            chord_progressions (List[List[str]]): Progresiones de acordes típicas.
        """
        self.name: str = name
        self.tempo_range: Tuple[int, int] = tempo_range
        self.key_signatures: List[str] = key_signatures
        self.time_signatures: List[str] = time_signatures
        self.chord_progressions: List[List[str]] = chord_progressions
    
    def get_random_tempo(self) -> int:
        """
        Obtener tempo aleatorio dentro del rango del género.
        
        Returns:
            int: Tempo en BPM.
        """
        return np.random.randint(self.tempo_range[0], self.tempo_range[1] + 1)
    
    def get_random_key(self) -> str:
        """
        Obtener tonalidad aleatoria del género.
        
        Returns:
            str: Tonalidad musical.
        """
        return np.random.choice(self.key_signatures)
    
    def get_random_chord_progression(self) -> List[str]:
        """
        Obtener progresión de acordes aleatoria.
        
        Returns:
            List[str]: Lista de acordes.
        """
        return np.random.choice(self.chord_progressions).tolist()


# Definiciones de géneros musicales
GENRES: Dict[str, MusicGenre] = {
    'rock': MusicGenre(
        name='Rock',
        tempo_range=(120, 140),
        key_signatures=['C', 'G', 'D', 'A', 'E'],
        time_signatures=['4/4'],
        chord_progressions=[
            ['I', 'IV', 'V', 'I'],
            ['I', 'V', 'vi', 'IV'],
            ['vi', 'IV', 'I', 'V']
        ]
    ),
    
    'jazz': MusicGenre(
        name='Jazz',
        tempo_range=(60, 120),
        key_signatures=['C', 'F', 'Bb', 'Eb'],
        time_signatures=['4/4', '3/4'],
        chord_progressions=[
            ['ii', 'V', 'I'],
            ['iii', 'VI', 'ii', 'V', 'I'],
            ['I', 'vi', 'ii', 'V']
        ]
    ),
    
    'classical': MusicGenre(
        name='Classical',
        tempo_range=(40, 180),
        key_signatures=['C', 'G', 'D', 'A', 'F', 'Bb', 'Eb'],
        time_signatures=['4/4', '3/4', '6/8'],
        chord_progressions=[
            ['I', 'IV', 'V', 'I'],
            ['I', 'vi', 'IV', 'V'],
            ['I', 'V', 'vi', 'iii', 'IV', 'I', 'IV', 'V']
        ]
    ),
    
    'pop': MusicGenre(
        name='Pop',
        tempo_range=(100, 130),
        key_signatures=['C', 'G', 'D', 'A'],
        time_signatures=['4/4'],
        chord_progressions=[
            ['I', 'V', 'vi', 'IV'],
            ['I', 'vi', 'IV', 'V'],
            ['vi', 'IV', 'I', 'V']
        ]
    ),
    
    'electronic': MusicGenre(
        name='Electronic',
        tempo_range=(128, 140),
        key_signatures=['C', 'Am', 'G', 'Em'],
        time_signatures=['4/4'],
        chord_progressions=[
            ['i', 'VII', 'VI', 'V'],
            ['i', 'VI', 'III', 'VII'],
            ['I', 'V', 'VI', 'III']
        ]
    )
}


class MusicDataProcessor:
    """
    Clase para procesamiento de datos musicales.
    
    Aplica el Principio de Responsabilidad Única: Solo se encarga de
    preprocesar datos musicales para los modelos.
    """
    
    def __init__(self, sequence_length: int = 32, feature_dim: int = 128) -> None:
        """
        Inicializar el procesador de datos.
        
        Args:
            sequence_length (int): Longitud de secuencias.
            feature_dim (int): Dimensión de características.
        """
        self.sequence_length: int = sequence_length
        self.feature_dim: int = feature_dim
        self.midi_min: int = 21  # A0
        self.midi_max: int = 108  # C8
    
    def midi_to_features(self, midi_path: str) -> np.ndarray:
        """
        Convertir archivo MIDI a características numéricas.
        
        Args:
            midi_path (str): Ruta del archivo MIDI.
            
        Returns:
            np.ndarray: Características extraídas.
        """
        try:
            # Cargar archivo MIDI
            y, sr = librosa.load(midi_path, sr=None)
            
            # Extraer características
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            
            # Combinar características
            features = np.vstack([
                chroma,
                mfcc,
                spectral_centroid
            ])
            
            # Normalizar y padding
            features = self._normalize_features(features)
            features = self._pad_or_truncate(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error procesando MIDI {midi_path}: {str(e)}")
            return self._create_dummy_features()
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalizar características.
        
        Args:
            features (np.ndarray): Características originales.
            
        Returns:
            np.ndarray: Características normalizadas.
        """
        return (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-8)
    
    def _pad_or_truncate(self, features: np.ndarray) -> np.ndarray:
        """
        Ajustar características a la longitud deseada.
        
        Args:
            features (np.ndarray): Características originales.
            
        Returns:
            np.ndarray: Características ajustadas.
        """
        if features.shape[1] > self.sequence_length:
            return features[:, :self.sequence_length]
        elif features.shape[1] < self.sequence_length:
            padding = np.zeros((features.shape[0], self.sequence_length - features.shape[1]))
            return np.hstack([features, padding])
        return features
    
    def _create_dummy_features(self) -> np.ndarray:
        """
        Crear características dummy para fallback.
        
        Returns:
            np.ndarray: Características dummy.
        """
        return np.random.randn(self.feature_dim, self.sequence_length) * 0.1
    
    def features_to_midi(self, features: np.ndarray, output_path: str, 
                       tempo: int = 120) -> None:
        """
        Convertir características a archivo MIDI.
        
        Args:
            features (np.ndarray): Características generadas.
            output_path (str): Ruta de salida MIDI.
            tempo (int): Tempo en BPM.
        """
        try:
            # Convertir características a notas MIDI
            midi_notes = self._features_to_notes(features)
            
            # Crear archivo MIDI
            from midiutil import MIDIFile
            
            midi = MIDIFile(1)
            midi.addTempo(0, 0, tempo)
            
            time = 0
            for note, duration in midi_notes:
                midi.addNote(0, 0, note, time, duration, 100)
                time += duration
            
            # Guardar archivo
            with open(output_path, 'wb') as output_file:
                midi.writeFile(output_file)
            
            logger.info(f"Archivo MIDI guardado: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creando MIDI: {str(e)}")
    
    def _features_to_notes(self, features: np.ndarray) -> List[Tuple[int, int]]:
        """
        Convertir características a notas MIDI.
        
        Args:
            features (np.ndarray): Características generadas.
            
        Returns:
            List[Tuple[int, int]]: Lista de (nota, duración).
        """
        notes = []
        
        for i in range(0, features.shape[1], 2):  # Cada 2 pasos
            # Obtener nota más probable
            note_probs = features[:self.midi_max - self.midi_min, i]
            if len(note_probs) > 0:
                note_idx = np.argmax(note_probs)
                note = note_idx + self.midi_min
                
                # Duración basada en la siguiente característica
                duration = max(0.5, min(2.0, features[0, min(i + 1, features.shape[1] - 1)]))
                
                notes.append((int(note), float(duration)))
        
        return notes


class MusicGenerator:
    """
    Clase principal para generación de música.
    
    Aplica el Principio de Responsabilidad Única: Solo se encarga de
    generar música usando modelos RNN/LSTM.
    
    Attributes:
        model (Optional[tf.keras.Model]): Modelo LSTM entrenado.
        data_processor (MusicDataProcessor): Procesador de datos.
        genre (MusicGenre): Género musical actual.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 sequence_length: int = 32, feature_dim: int = 128) -> None:
        """
        Inicializar el generador musical.
        
        Args:
            model_path (Optional[str]): Ruta al modelo pre-entrenado.
            sequence_length (int): Longitud de secuencias.
            feature_dim (int): Dimensión de características.
        """
        self.model: Optional[tf.keras.Model] = None
        self.data_processor: MusicDataProcessor = MusicDataProcessor(sequence_length, feature_dim)
        self.genre: Optional[MusicGenre] = None
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self._create_demo_model()
    
    def _create_demo_model(self) -> None:
        """Crear modelo de demostración básico."""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(256, return_sequences=True, 
                                   input_shape=(32, 128)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(256, return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(88, activation='softmax')  # 88 teclas de piano
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            logger.info("Modelo de demostración creado exitosamente")
            
        except Exception as e:
            logger.error(f"Error creando modelo: {str(e)}")
    
    def load_model(self, model_path: str) -> None:
        """
        Cargar modelo pre-entrenado.
        
        Args:
            model_path (str): Ruta del modelo.
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Modelo cargado desde: {model_path}")
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            self._create_demo_model()
    
    def set_genre(self, genre_name: str) -> None:
        """
        Establecer género musical.
        
        Args:
            genre_name (str): Nombre del género.
        """
        if genre_name in GENRES:
            self.genre = GENRES[genre_name]
            logger.info(f"Género establecido: {genre_name}")
        else:
            logger.error(f"Género no soportado: {genre_name}")
            logger.info(f"Géneros disponibles: {list(GENRES.keys())}")
    
    def generate(self, length: int = 120, temperature: float = 0.8,
                seed_sequence: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generar secuencia musical.
        
        Args:
            length (int): Longitud en pasos temporales.
            temperature (float): Temperatura de generación.
            seed_sequence (Optional[np.ndarray]): Secuencia inicial.
            
        Returns:
            np.ndarray: Secuencia musical generada.
        """
        if self.model is None:
            logger.error("Modelo no disponible")
            return np.random.randn(128, length)
        
        try:
            # Inicializar secuencia
            if seed_sequence is None:
                current_sequence = np.random.randn(1, 32, 128) * 0.1
            else:
                current_sequence = seed_sequence.reshape(1, 32, 128)
            
            generated = []
            
            for _ in range(length):
                # Predecir siguiente paso
                predictions = self.model.predict(current_sequence, verbose=0)
                
                # Aplicar temperatura y samplear
                predictions = predictions[0, -1, :] / temperature
                predictions = np.exp(predictions - np.max(predictions))
                predictions = predictions / np.sum(predictions)
                
                next_step = np.random.choice(len(predictions), p=predictions)
                next_feature = np.zeros(128)
                next_feature[next_step] = 1.0
                
                generated.append(next_feature)
                
                # Actualizar secuencia
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, :] = next_feature
            
            return np.array(generated).T
            
        except Exception as e:
            logger.error(f"Error en generación: {str(e)}")
            return np.random.randn(128, length)
    
    def generate_song(self, genre: str, sections: List[str], 
                    total_length: int = 180) -> Dict[str, Any]:
        """
        Generar canción completa con múltiples secciones.
        
        Args:
            genre (str): Género musical.
            sections (List[str]): Secciones de la canción.
            total_length (int): Duración total en segundos.
            
        Returns:
            Dict[str, Any]: Información de la canción generada.
        """
        self.set_genre(genre)
        
        if self.genre is None:
            return {'error': 'Género no válido'}
        
        # Calcular duración por sección
        section_length = total_length // len(sections)
        
        song_data = {
            'genre': genre,
            'tempo': self.genre.get_random_tempo(),
            'key': self.genre.get_random_key(),
            'time_signature': np.random.choice(self.genre.time_signatures),
            'chord_progression': self.genre.get_random_chord_progression(),
            'sections': {}
        }
        
        # Generar cada sección
        for section in sections:
            section_data = self.generate(
                length=section_length * 2,  # Aproximación
                temperature=0.8
            )
            
            song_data['sections'][section] = {
                'features': section_data,
                'duration': section_length
            }
        
        return song_data
    
    def save_midi(self, music_data: np.ndarray, output_path: str, 
                  tempo: int = 120) -> None:
        """
        Guardar secuencia musical como archivo MIDI.
        
        Args:
            music_data (np.ndarray): Datos musicales.
            output_path (str): Ruta de salida.
            tempo (int): Tempo en BPM.
        """
        self.data_processor.features_to_midi(music_data, output_path, tempo)
    
    def save_song(self, song_data: Dict[str, Any], output_dir: str) -> None:
        """
        Guardar canción completa en múltiples formatos.
        
        Args:
            song_data (Dict[str, Any]): Datos de la canción.
            output_dir (str): Directorio de salida.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        song_name = f"{song_data['genre']}_song_{song_data['key']}"
        
        # Guardar metadatos
        metadata = {
            'genre': song_data['genre'],
            'tempo': song_data['tempo'],
            'key': song_data['key'],
            'time_signature': song_data['time_signature'],
            'chord_progression': song_data['chord_progression'],
            'sections': list(song_data['sections'].keys())
        }
        
        with open(output_path / f"{song_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Guardar cada sección como MIDI
        for section_name, section_data in song_data['sections'].items():
            section_path = output_path / f"{song_name}_{section_name}.mid"
            self.save_midi(section_data['features'], str(section_path), song_data['tempo'])
        
        # Combinar todas las secciones
        combined_features = np.hstack([
            section['features'] for section in song_data['sections'].values()
        ])
        
        combined_path = output_path / f"{song_name}_complete.mid"
        self.save_midi(combined_features, str(combined_path), song_data['tempo'])
        
        logger.info(f"Canción guardada en: {output_path}")


def main() -> None:
    """
    Función principal para demostración del generador musical.
    """
    logger.info("🎵 Iniciando Generador Musical")
    
    # Crear directorios necesarios
    Path('models').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    
    # Inicializar generador
    generator = MusicGenerator()
    
    # Menú de opciones
    print("\n🎵 Generador Musical de IA")
    print("=" * 40)
    print("1. Generar música por género")
    print("2. Generar canción completa")
    print("3. Entrenar modelo")
    print("4. Salir")
    
    choice = input("\nSelecciona una opción (1-4): ").strip()
    
    if choice == '1':
        # Generar música por género
        genre = input("Género (rock/jazz/classical/pop/electronic): ").strip().lower()
        length = int(input("Longitud en segundos: "))
        
        generator.set_genre(genre)
        if generator.genre:
            music = generator.generate(length=length * 2)  # Aproximación
            output_path = f"outputs/{genre}_music.mid"
            generator.save_midi(music, output_path)
            print(f"✅ Música guardada en: {output_path}")
    
    elif choice == '2':
        # Generar canción completa
        genre = input("Género: ").strip().lower()
        sections_input = input("Secciones (ej: intro,verse,chorus): ").strip()
        sections = [s.strip() for s in sections_input.split(',')]
        total_length = int(input("Duración total (segundos): "))
        
        song = generator.generate_song(genre, sections, total_length)
        generator.save_song(song, 'outputs')
        print("✅ Canción completa generada")
    
    elif choice == '3':
        # Entrenar modelo (placeholder)
        print("🔄 Función de entrenamiento en desarrollo...")
    
    elif choice == '4':
        logger.info("👋 Saliendo del generador musical...")
    else:
        logger.error("❌ Opción no válida")


if __name__ == "__main__":
    main()
