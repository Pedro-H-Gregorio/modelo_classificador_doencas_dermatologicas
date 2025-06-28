# Projeto de Inteligência Artificial - Dermatologia

Este projeto realiza o treinamento e a predição de modelos de classificação para dados dermatológicos, utilizando dados tabulares e de imagem. O pipeline inclui pré-processamento, treinamento, avaliação e salvamento do modelo e pipeline.

---

## Pré-requisitos

- Python 3.10+
- [pip](https://pip.pypa.io/en/stable/)
- [virtualenv](https://virtualenv.pypa.io/en/latest/) (opcional, mas recomendado)

---

## Instalação

1. **Crie e ative um ambiente virtual (opcional, mas recomendado):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuração

Faça o download do dataset e salve ele no diretório que foi clonado. Segue o link do dataset:
[Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

Crie um arquivo .env na pasta config com as seguintes variáveis (ajuste os caminhos conforme necessário):

```
PROJECT_ROOT_PATH=/caminho/absoluto/para/projeto-5/src
MODEL=svc
DATA_PATH=/caminho/absoluto/para/seus/dados
```

- `MODEL` pode ser `svc` (classificação) ou `one_class_svm` (detecção de anomalia).
- `DATA_PATH` deve apontar para a pasta onde estão os dados de treino.

---

## Como gerar o modelo

Execute o script de geração do modelo:

```bash
python generate.py
```

O pipeline irá:

- Processar os dados
- Treinar o modelo
- Avaliar o desempenho
- Salvar o modelo, pipeline e metadados nas pastas `models/` e `pipelines/` dentro do diretório do projeto

---

## Como rodar a predição

Use o script main.py para realizar uma predição. Exemplo de uso:

```bash
python main.py <metadata_model_path> <age> <sex> <localization> <image_path>
```

**Exemplo:**

```bash
python main.py models/dermatology_model_svc.pkl 45 male back images/teste.jpg
```

- `<model_path>`: Caminho para o arquivo do modelo treinado (ex: `models/dermatology_model_svc.pkl`)
- `<age>`: Idade do paciente (ex: `45`)
- `<sex>`: Sexo (`male` ou `female`)
- `<localization>`: Localização da lesão (ex: `back`)
- `<image_path>`: Caminho para a imagem da lesão

O resultado da predição será exibido no console.

---

## Estrutura dos arquivos gerados

- `models/dermatology_model_<tipo>.pkl`: Modelo treinado
- `models/metadata_dermatology_model_<tipo>.json`: Metadados do modelo
- `pipelines/pipeline_<tipo>.pkl`: Pipeline de pré-processamento

---

## Observações

- Certifique-se de que os caminhos no .env estão corretos.
- Os logs do processo são exibidos no console.
- Para dúvidas sobre os parâmetros ou funcionamento, consulte os comentários no código ou abra uma issue.

---

**Bom uso!**
