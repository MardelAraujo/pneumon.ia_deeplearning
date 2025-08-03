import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf

def evaluate_and_report(model: tf.keras.Model, test_generator):
    """
    Avalia o modelo e imprime o relatório de classificação e a matriz de confusão.

    Args:
        model (tf.keras.Model): O modelo treinado.
        test_generator: Gerador de dados de teste.
    """
    print("\nAvaliando modelo no conjunto de teste...")
    test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
    print(f'Acurácia no conjunto de teste: {test_acc * 100:.2f}%')

    # Obter rótulos verdadeiros e preditos
    test_labels = test_generator.classes
    # Fazer predições no conjunto de teste
    predictions = model.predict(test_generator, steps=len(test_generator))
    # Converter probabilidades para classes binárias (0 ou 1)
    predicted_labels = (predictions > 0.5).astype(int)

    # Matriz de Confusão
    conf_matrix = confusion_matrix(test_labels, predicted_labels)
    print("\nMatriz de Confusão:")
    print(conf_matrix)

    # Relatório de Classificação (Precissão, Recall, F1-Score)
    class_report = classification_report(test_labels, predicted_labels, target_names=['Normal', 'Pneumonia'])
    print("\nRelatório de Classificação:")
    print(class_report)

    # Plotagem da Matriz de Confusão para melhor visualização
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Normal', 'Pneumonia'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.show()

def plot_learning_curves(history):
    """
    Plota as curvas de acurácia e perda (loss) do histórico de treinamento.

    Args:
        history (tf.keras.callbacks.History): Objeto History do treinamento do modelo.
    """
    print("\nPlotando curvas de aprendizado...")
    plt.figure(figsize=(12, 4))

    # Plotagem da Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Acurácia no Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia na Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.title('Acurácia por Época')
    plt.legend()

    # Plotagem da Perda (Loss)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Perda no Treinamento')
    plt.plot(history.history['val_loss'], label='Perda na Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.title('Perda por Época')
    plt.legend()

    plt.tight_layout() 
    plt.show()