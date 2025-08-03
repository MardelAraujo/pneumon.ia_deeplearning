import yaml
import os
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.data_loader import get_data_generators
from src.model_builder import build_pneumonia_model, fine_tune_model
from src.trainer import train_model
from src.evaluator import evaluate_and_report, plot_learning_curves

def set_all_seeds(seed: int = 42):
    """Define seeds para reprodutibilidade."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seeds definidas para {seed} para reprodutibilidade.")

if __name__ == "__main__":
    # Carregar Configurações
    # Abre o arquivo config.yaml e carrega os parâmetros definidos.
    print("Carregando configurações do 'config/config.yaml'...")
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("Configurações carregadas com sucesso.")
    except FileNotFoundError:
        print("ERRO: Arquivo 'config/config.yaml' não encontrado. Certifique-se de que ele existe.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"ERRO: Problema ao carregar o arquivo YAML: {e}")
        exit(1)

    set_all_seeds(config.get('seed', 42)) 

    # Prepara Caminhos dos Dados
    
    base_data_dir = config['data_paths']['base_data_dir']
    extracted_full_path = os.path.join(base_data_dir, config['data_paths']['kaggle_dataset_name'].split('/')[1])
    train_dir = os.path.join(extracted_full_path, config['data_paths']['extracted_sub_dir_name'], config['data_paths']['train_sub_dir'])
    test_dir = os.path.join(extracted_full_path, config['data_paths']['extracted_sub_dir_name'], config['data_paths']['test_sub_dir'])

    # Verifica se os diretórios de dados existem
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"ERRO: Diretórios de treinamento/teste não encontrados em '{train_dir}' e '{test_dir}'.")
        print("Por favor, execute 'python scripts/download_data.py' primeiro para baixar e extrair o dataset.")
        exit(1)

    # Obter Geradores de Dados
    train_generator, test_generator = get_data_generators(
        train_dir,
        test_dir,
        target_size=tuple(config['model_params']['target_size']),
        batch_size=config['training_params']['batch_size']
    )

    # Constroi Modelo
    # Chama a função do módulo model_builder para criar o modelo CNN.
    model, base_model = build_pneumonia_model(
        input_shape=tuple(config['model_params']['input_shape']),
        dense_units=config['model_params']['dense_units'],
        dropout_rate=config['model_params']['dropout_rate'],
        learning_rate=config['training_params']['initial_learning_rate'] # Passa a LR inicial
    )

    # Define Callbacks de Treinamento
    callbacks = [
        EarlyStopping(
            monitor=config['callbacks']['early_stopping_monitor'],
            patience=config['callbacks']['early_stopping_patience'],
            restore_best_weights=config['callbacks']['early_stopping_restore_best_weights']
        ),
        ReduceLROnPlateau(
            monitor=config['callbacks']['reduce_lr_monitor'],
            factor=config['callbacks']['reduce_lr_factor'],
            patience=config['callbacks']['reduce_lr_patience'],
            min_lr=config['callbacks']['reduce_lr_min_lr']
        )
    ]

    # Treinamento Inicial (Camadas Base Congeladas)
    print("\n--- Iniciando Treinamento Inicial (camadas base do VGG16 congeladas) ---")
    history_initial = train_model(
        model,
        train_generator,
        test_generator,
        epochs=config['training_params']['epochs'],
        learning_rate=config['training_params']['initial_learning_rate'],
        callbacks=callbacks
    )

    # Fine-tuning do Modelo
    # Se o fine-tuning for configurado, descongela as últimas camadas do modelo base e continua o treinamento.
    if config['training_params'].get('fine_tune_epochs', 0) > 0:
        fine_tune_epochs = config['training_params']['fine_tune_epochs']
        fine_tune_lr = config['training_params']['fine_tune_learning_rate']
        unfreeze_layers = config['training_params']['fine_tune_unfreeze_layers']

        fine_tune_model(model, base_model, unfreeze_layers, fine_tune_lr)

        print(f"\n--- Continuando Treinamento (Fine-tuning por {fine_tune_epochs} épocas) ---")
        history_fine_tune = train_model(
            model,
            train_generator,
            test_generator,
            epochs=fine_tune_epochs,
            learning_rate=fine_tune_lr,
            callbacks=callbacks
        )
        for key in history_fine_tune.history:
            history_initial.history[key] += history_fine_tune.history[key]

    # Avaliação Final do Modelo
    # Chama a função do módulo evaluator para avaliar e gerar relatórios.
    print("\n--- Avaliação Final do Modelo ---")
    evaluate_and_report(model, test_generator)

    # Plotagem das Curvas de Aprendizado
    # Chama a função do módulo evaluator para plotar as curvas de acurácia e perda.
    plot_learning_curves(history_initial)

    # Salva Modelo Treinado

    model_save_path = config['model_save_dir']
    os.makedirs(model_save_path, exist_ok=True) 
    model.save(os.path.join(model_save_path, 'pneumonia_classifier_model.tf'), save_format='tf')
    print(f"\nModelo final salvo em: {os.path.join(model_save_path, 'pneumonia_classifier_model.tf')}")