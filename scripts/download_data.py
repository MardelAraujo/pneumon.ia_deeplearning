import os
import subprocess
import zipfile
import yaml 

def download_and_extract_kaggle_dataset(config_path='config/config.yaml'):
    """
    Baixa e extrai o dataset do Kaggle para o diretório de dados especificado.

    Args:
        config_path (str): Caminho para o arquivo de configuração YAML.
    """
    print("Iniciando download e extração de dados do Kaggle...")

    # Carregar configurações
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data_paths']
    base_data_dir = data_config['base_data_dir']
    kaggle_dataset_name = data_config['kaggle_dataset_name']
    zip_file_name = data_config['zip_file_name']
    extracted_sub_dir_name = data_config['extracted_sub_dir_name']

    # Criar diretório base de dados se não existir
    os.makedirs(base_data_dir, exist_ok=True)
    print(f"Diretório base de dados criado/verificado: {base_data_dir}")

    # Caminho completo para o arquivo ZIP baixado
    zip_file_path = os.path.join(base_data_dir, zip_file_name)
    # Caminho completo para o diretório onde o conteúdo será extraído
    extracted_dir_path = os.path.join(base_data_dir, extracted_sub_dir_name)

    # Verificar se o dataset já foi baixado e extraído
    if os.path.exists(os.path.join(extracted_dir_path, data_config['train_sub_dir'])):
        print(f"Dataset já extraído em {extracted_dir_path}. Pulando download e extração.")
        return extracted_dir_path 

    # Baixar dataset do Kaggle

    print(f"Baixando dataset '{kaggle_dataset_name}' para {base_data_dir}...")
    download_command = ['kaggle', 'datasets', 'download', '-d', kaggle_dataset_name, '-p', base_data_dir]
    try:
        # subprocess.run executa comandos externos. 
        # check=True levanta um erro se o comando falhar.
        subprocess.run(download_command, check=True)
        print(f"Dataset '{kaggle_dataset_name}' baixado com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"ERRO: Não foi possível baixar o dataset do Kaggle. Por favor, verifique:")
        print(f"  - Se 'kaggle.json' está em {os.path.expanduser('~/.kaggle/')}.")
        print(f"  - Se as permissões de 'kaggle.json' estão corretas (chmod 600).")
        print(f"  - Se você tem o cliente Kaggle instalado e configurado corretamente.")
        print(f"Detalhes do erro: {e}")
        exit(1) 

    # Extrair o arquivo ZIP 
    print(f"Extraindo '{zip_file_name}' para '{extracted_dir_path}'...")
    try:
        # Verifica se o arquivo ZIP existe
        if not os.path.exists(zip_file_path):
            raise FileNotFoundError(f"Arquivo ZIP não encontrado em: {zip_file_path}")

        # Cria o diretório de extração se não existir
        os.makedirs(extracted_dir_path, exist_ok=True)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir_path)
        print("Extração concluída com sucesso.")

        # Remover o arquivo ZIP após a extração para economizar espaço
        os.remove(zip_file_path)
        print(f"Arquivo ZIP '{zip_file_name}' removido.")

    except zipfile.BadZipFile:
        print(f"ERRO: O arquivo '{zip_file_name}' está corrompido ou não é um ZIP válido.")
        exit(1)
    except FileNotFoundError as e:
        print(f"ERRO: {e}")
        exit(1)
    except Exception as e:
        print(f"Ocorreu um erro durante a extração: {e}")
        exit(1)

    return extracted_dir_path 

if __name__ == '__main__':
    # Este bloco só é executado se você rodar download_data.py diretamente
    download_and_extract_kaggle_dataset()
    print("Processo de download e extração finalizado.")