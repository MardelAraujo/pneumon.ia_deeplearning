import tensorflow as tf

def train_model(model: tf.keras.Model, train_generator, validation_generator, epochs: int, learning_rate: float, callbacks: list = None):
    """
    Realiza o treinamento do modelo.

    Args:
        model (tf.keras.Model): O modelo compilado para treinamento.
        train_generator: Gerador de dados de treinamento.
        validation_generator: Gerador de dados de validação.
        epochs (int): Número de épocas para o treinamento.
        learning_rate (float): Taxa de aprendizado para o otimizador (pode ser usado para recompilar se o otimizador for redefinido).
        callbacks (list): Lista de callbacks do Keras a serem usados (ex: EarlyStopping, ReduceLROnPlateau).

    Returns:
        tf.keras.callbacks.History: Objeto History contendo o histórico de treinamento.
    """
    print(f"\nIniciando treinamento do modelo por {epochs} épocas...")
    print(f"Taxa de aprendizado: {learning_rate}")

    # Garante que o otimizador tenha a taxa de aprendizado correta se for a primeira compilação ou recompilação
   
    model.optimizer.learning_rate.assign(learning_rate)

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks
    )
    print("Treinamento concluído.")
    return history