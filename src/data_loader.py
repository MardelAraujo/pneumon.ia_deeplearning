import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(train_dir: str, test_dir: str, target_size: tuple, batch_size: int):
    """
    Cria e retorna os geradores de dados para treinamento e teste.

    Args:
        train_dir (str): Caminho para o diretório de treinamento.
        test_dir (str): Caminho para o diretório de teste.
        target_size (tuple): Tamanho (altura, largura) para redimensionar as imagens.
        batch_size (int): Tamanho do batch para os geradores.

    Returns:
        tuple: (train_generator, test_generator)
    """
    print(f"Configurando geradores de dados...")

    # Aumento de dados para o conjunto de treinamento
    # Aplica transformações para variar as imagens e melhorar a generalização do modelo.
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,        # Normaliza os pixels para [0, 1]
        rotation_range=20,          # Rotaciona imagens em até 20 graus
        width_shift_range=0.2,      # Desloca horizontalmente
        height_shift_range=0.2,     # Desloca verticalmente
        shear_range=0.2,            # Aplica cisalhamento
        zoom_range=0.2,             # Aplica zoom
        horizontal_flip=True,       # Vira imagens horizontalmente
        fill_mode='nearest'         # Preenche pixels extras
    )

    # Carregamento de imagens de treinamento
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',        # Classificação binária (Normal, Pneumonia)
        shuffle=True                # Embaralha os dados de treinamento
    )
    print(f"Gerador de treinamento configurado: Encontradas {train_generator.samples} imagens em {train_generator.num_classes} classes.")

    # Gerador para o conjunto de teste 
    # (Não aplicamos aumento de dados ao conjunto de teste para não distorcer a avaliação.)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # Carregamento de imagens de teste
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False               # Não embaralha os dados de teste para avaliação consistente
    )
    print(f"Gerador de teste configurado: Encontradas {test_generator.samples} imagens em {test_generator.num_classes} classes.")

    return train_generator, test_generator