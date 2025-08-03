import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

def build_pneumonia_model(input_shape: tuple, dense_units: int, dropout_rate: float, learning_rate: float):
    """
    Constrói e compila o modelo de CNN usando VGG16 para Transfer Learning.

    Args:
        input_shape (tuple): Formato da imagem de entrada (altura, largura, canais).
        dense_units (int): Número de neurônios na camada densa oculta.
        dropout_rate (float): Taxa de dropout para a camada de Dropout.
        learning_rate (float): Taxa de aprendizado para o otimizador Adam.

    Returns:
        tuple: (model, base_model) - o modelo compilado e o modelo base VGG16.
    """
    print(f"Construindo modelo VGG16 com camadas personalizadas...")

    # Carrega o modelo VGG16 pré-treinado no ImageNet
    # include_top=False: remove as camadas de classificação finais do VGG16.
    # input_shape: define o formato das imagens que entrarão no modelo.
    # O VGG16 espera imagens de tamanho 224x224 com 3 canais (RGB).
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Congela as camadas do modelo base VGG16.

    base_model.trainable = False

    # Modelo Sequencial

    model = Sequential()
    model.add(base_model)      
    model.add(Flatten())       
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate)) 
    model.add(Dense(1, activation='sigmoid')) 

    # Compila o modelo: define o otimizador, a função de perda e as métricas.
    # loss='binary_crossentropy': função de perda adequada para classificação binária.
    # metrics=['accuracy']: métrica para acompanhar o desempenho (acurácia).

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary() # Imprime um resumo da arquitetura do modelo
    return model, base_model

def fine_tune_model(model: tf.keras.Model, base_model: tf.keras.Model, num_unfreeze_layers: int, fine_tune_learning_rate: float):
    """
    Configura o modelo para fine-tuning, descongelando as últimas camadas do modelo base.

    Args:
        model (tf.keras.Model): O modelo já treinado com as camadas base congeladas.
        base_model (tf.keras.Model): O modelo base (VGG16).
        num_unfreeze_layers (int): Número de camadas do final do base_model a serem descongeladas.
        fine_tune_learning_rate (float): Taxa de aprendizado para o fine-tuning.
    """
    print(f"\nConfigurando modelo para Fine-tuning: Descongelando as últimas {num_unfreeze_layers} camadas do VGG16.")
    base_model.trainable = True 

    # Congela novamente as primeiras camadas do VGG16, deixando apenas as últimas 'num_unfreeze_layers' treináveis.
    for layer in base_model.layers[:-num_unfreeze_layers]:
        layer.trainable = False

    # Recompila o modelo com uma taxa de aprendizado muito menor.
    # Uma taxa menor é crucial para o fine-tuning para não "queimar" os pesos pré-treinados.
    model.compile(optimizer=Adam(learning_rate=fine_tune_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()