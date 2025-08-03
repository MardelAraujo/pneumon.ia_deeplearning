# 🩺 Deep Learning Pneumonia Classifier

## 📋 Objetivo do Projeto

Este projeto tem como objetivo desenvolver um classificador de pneumonia em radiografias de tórax utilizando técnicas de Deep Learning e Transfer Learning (VGG16). O sistema automatiza o download do dataset, o pré-processamento das imagens, o treinamento do modelo e a avaliação dos resultados.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.10+**
- **TensorFlow / Keras** (Deep Learning)
- **NumPy** (Manipulação de arrays)
- **PyYAML** (Leitura de arquivos de configuração)
- **Kaggle API** (Download automático do dataset)
- **Matplotlib** (Visualização de resultados)

---

## 📁 Estrutura do Projeto

```
projetos/deep_learn_pneumonia/
│
├── config/
│   └── config.yaml           # Configurações do projeto
├── data/                      # Dados brutos e processados
├── models/                    # Modelos treinados
├── scripts/
│   ├── download_data.py       # Script para baixar e extrair o dataset
│   └── train_model.py         # Script principal de treinamento
├── src/
│   ├── data_loader.py         # Funções para carregar e preparar dados
│   ├── model_builder.py       # Construção e fine-tuning do modelo
│   ├── trainer.py             # Funções de treinamento
│   └── evaluator.py           # Avaliação do modelo
├── utils/
│   └── visualization.py       # Visualização de métricas e resultados
└── requirements.txt           # Dependências do projeto
```

---

## 🚀 Como Rodar na Sua Máquina

### 1. Clone o repositório
```bash
git clone <url-do-repositorio>
cd deep_learn_pneumonia
```

### 2. Crie e ative um ambiente virtual
```bash
python -m venv .venv
# Ative o ambiente:
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Configure a API do Kaggle
- Crie uma conta em https://www.kaggle.com/ ➡️ "My Account" ➡️ "Create New API Token".
- Baixe o arquivo `kaggle.json`.
- Coloque o arquivo em `C:\Users\<SeuUsuario>\.kaggle\kaggle.json` (Windows) ou `~/.kaggle/kaggle.json` (Linux/Mac).
- Dê permissão ao arquivo:
  - Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`

### 5. Baixe e extraia o dataset automaticamente
```bash
python scripts/download_data.py
```

### 6. Treine o modelo
```bash
# Execute a partir da raiz do projeto:
$env:PYTHONPATH="."; python scripts/train_model.py  # Windows PowerShell
# Ou
PYTHONPATH=. python scripts/train_model.py           # Linux/Mac
```

---

## ⚙️ Configuração

Todas as configurações (caminhos, hiperparâmetros, callbacks) estão no arquivo `config/config.yaml`. Edite conforme necessário para ajustar o experimento.

---

## 📊 Resultados e Visualização

- Os modelos treinados são salvos em `models/`.
- Métricas e gráficos podem ser gerados com scripts em `utils/visualization.py`.

---

## 🧩 Fluxo do Projeto

1. **Download dos dados**
2. **Pré-processamento e aumento de dados**
3. **Construção do modelo (VGG16 + camadas customizadas)**
4. **Treinamento inicial (camadas base congeladas)**
5. **Fine-tuning (descongelando camadas finais)**
6. **Avaliação e visualização dos resultados**

---

## 📝 Observações Importantes

- O projeto foi desenvolvido para classificação binária (Pneumonia vs. Normal).
- O dataset utilizado é o [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
- Para reprodutibilidade, seeds são definidas no início do treinamento.
- O código é modular e fácil de adaptar para outros datasets de imagens médicas.

---

## 💡 Dicas

- Se encontrar problemas de importação, sempre execute os scripts a partir da raiz do projeto e configure o `PYTHONPATH`.
- Para treinar em GPU, certifique-se de ter o TensorFlow com suporte a CUDA instalado.
- Consulte e ajuste o `config.yaml` para experimentar diferentes hiperparâmetros.

---

## 👨‍💻 Autor

- Projeto criado por Mardel Araújo.
- Contato: mardelaraujo1@gmailcom

---

## 🏥 Bons estudos e ótimos experimentos! 🚑📈
