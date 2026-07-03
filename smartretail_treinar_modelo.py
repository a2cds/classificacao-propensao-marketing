"""Script de Automação para Treinamento do Modelo de Propensão - SmartRetail.

Este arquivo contém o pipeline estruturado para carregamento de dados,
divisão de amostras, ajuste do classificador DecisionTree e avaliação de 
múltiplas métricas de performance. Projetado para integração a pipelines 
de agendamento (cron/Airflow) ou treinamento recorrente.
"""

import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuração de logging operacional do pipeline SmartRetail
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def carregar_dados_smartretail(caminho_csv: Path) -> pd.DataFrame:
    """Carrega a base histórica de marketing da SmartRetail.

    Args:
        caminho_csv (Path): Instância Path indicando o arquivo de dados.

    Raises:
        FileNotFoundError: Se a base de dados não for localizada na pasta data.

    Returns:
        pd.DataFrame: DataFrame com os dados brutos de clientes.
    """
    if not caminho_csv.exists():
        raise FileNotFoundError(f"Base de dados SmartRetail não localizada em: {caminho_csv.resolve()}")
    
    logging.info(f"Carregando histórico de campanhas SmartRetail de: {caminho_csv}")
    return pd.read_csv(caminho_csv)


def preparar_amostras_campanha(
    df: pd.DataFrame, 
    coluna_alvo: str, 
    test_size: float = 0.20, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Isola os dados descritivos e particiona em conjuntos de treino e teste.

    Args:
        df (pd.DataFrame): DataFrame bruto.
        coluna_alvo (str): Alvo preditivo ('RespondeuCampanha').
        test_size (float): Proporção para validação (padrão: 0.20).
        random_state (int): Semente para garantia de reprodutibilidade.

    Returns:
        Tuple: X_train, X_test, y_train, y_test.
    """
    logging.info(f"Separando atributos preditores da variável alvo '{coluna_alvo}'")
    X = df.drop(columns=[coluna_alvo])
    y = df[coluna_alvo]

    logging.info(f"Divisão estratificada: {100*(1-test_size):.0f}% Treino / {100*test_size:.0f}% Teste")
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def treinar_classificador_propensao(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = 42
) -> DecisionTreeClassifier:
    """Ajusta o modelo DecisionTree com limitação de profundidade.

    Args:
        X_train (pd.DataFrame): Dados de entrada para treino.
        y_train (pd.Series): Alvo correspondente para treino.
        random_state (int): Semente de aleatoriedade.

    Returns:
        DecisionTreeClassifier: Classificador treinado.
    """
    logging.info("Ajustando classificador DecisionTree com profundidade máxima mitigada...")
    # max_depth=3 limita a expansão da árvore para evitar sobreajuste nos dados históricos
    modelo = DecisionTreeClassifier(random_state=random_state, max_depth=3)
    modelo.fit(X_train, y_train)
    return modelo


def avaliar_previsoes_marketing(
    modelo: DecisionTreeClassifier, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> None:
    """Executa previsões no conjunto de teste e exibe o relatório de performance.

    Args:
        modelo (DecisionTreeClassifier): Modelo de propensão treinado.
        X_test (pd.DataFrame): Features de validação.
        y_test (pd.Series): Alvos reais de validação.
    """
    logging.info("Calculando predições e métricas de desempenho")
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)

    print("\n" + "="*60)
    print("MÉTRICAS DE PERFORMANCE: CLASSIFICADOR SMARTRETAIL")
    print("="*60)
    print(f"Acurácia Geral obtida em Teste: {acuracia * 100:.2f}%")
    
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(y_test, y_pred))
    print("="*60 + "\n")


def main() -> None:
    """Inicia a execução do pipeline de modelagem da SmartRetail."""
    caminho_dados = Path("data") / "clientes_marketing.csv"
    coluna_alvo = "RespondeuCampanha"

    try:
        df = carregar_dados_smartretail(caminho_dados)
        X_train, X_test, y_train, y_test = preparar_amostras_campanha(df, coluna_alvo)
        modelo = treinar_classificador_propensao(X_train, y_train)
        avaliar_previsoes_marketing(modelo, X_test, y_test)
    except FileNotFoundError as fnf_err:
        logging.error(fnf_err)
    except Exception as err:
        logging.critical(f"Falha de execução no pipeline: {err}", exc_info=True)


if __name__ == "__main__":
    main()