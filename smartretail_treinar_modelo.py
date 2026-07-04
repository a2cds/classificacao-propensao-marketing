"""
PROJETO: Modelo de Marketing de Propensão a Campanhas Promocionais
ALGORITMO: Árvore de Decisão (Decision Tree Classifier)
EMPRESA: SmartRetail

Este script executa o pipeline estruturado para treinamento e avaliação de um 
modelo preditivo de marketing de propensão. O objetivo é classificar a probabilidade 
de um cliente responder positivamente a uma nova campanha promocional. 
Ao final, o script gera e exporta a visualização lógica das regras de propensão 
da Árvore de Decisão diretamente para a pasta 'plots/'.
"""

import logging
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import pandas as pd

# Componentes de modelagem e avaliação do Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuração de logger operacional do pipeline de marketing preditivo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def carregar_dados_smartretail(caminho_csv: Path) -> pd.DataFrame:
    """Carrega o histórico de conversão em campanhas de marketing da SmartRetail.

    Args:
        caminho_csv (Path): Caminho para o arquivo CSV de entrada.

    Raises:
        FileNotFoundError: Se a base de dados não for localizada na pasta 'data'.

    Returns:
        pd.DataFrame: Histórico de clientes para modelagem de propensão.
    """
    if not caminho_csv.exists():
        raise FileNotFoundError(f"Dataset de marketing não localizado em: {caminho_csv.resolve()}")
    logging.info(f"Carregando histórico de campanhas de: {caminho_csv}")
    return pd.read_csv(caminho_csv)


def preparar_e_dividir_dados(
    df: pd.DataFrame, 
    coluna_alvo: str, 
    test_size: float = 0.20, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Isola os dados de consumo e divide de forma estratificada para modelagem de propensão.

    Garante que a proporção original de clientes que responderam (classe 1) e
    não responderam (classe 0) seja mantida nas duas partições.
    """
    X = df.drop(columns=[coluna_alvo])
    y = df[coluna_alvo]
    logging.info(f"Dividindo conjunto de dados de forma estratificada ({100*test_size:.0f}% teste)")
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def exportar_grafico_arvore_propensao(
    modelo_dt: DecisionTreeClassifier, 
    feature_names: list, 
    caminho_saida: Path
) -> None:
    """Gera e exporta as regras visuais da Árvore de Decisão de Propensão à Campanha."""
    logging.info("Renderizando o fluxograma lógico de decisões do modelo de propensão...")
    plt.figure(figsize=(12, 8))
    plot_tree(
        modelo_dt,
        feature_names=feature_names,
        class_names=['Não Respondeu (0)', 'Respondeu (1)'],
        filled=True,
        rounded=True,
        fontsize=9
    )
    plt.title("Visualização das Decisões de Classificação - SmartRetail (Propensão a Campanha)", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150)
    plt.close()
    logging.info(f"Gráfico de propensão de marketing exportado com sucesso para: {caminho_saida}")


def main() -> None:
    # Definição dos caminhos conforme a arquitetura padrão do projeto
    caminho_dados = Path("data") / "clientes_marketing.csv"
    coluna_alvo = "RespondeuCampanha"
    seed = 42

    # Configuração da pasta de destino recomendada para os gráficos gerados
    diretorio_plots = Path("plots")
    diretorio_plots.mkdir(exist_ok=True)
    caminho_img_dt = diretorio_plots / "smartretail_decision_tree.png"

    try:
        # 1. Carregamento dos dados históricos
        df = carregar_dados_smartretail(caminho_dados)
        X_train, X_test, y_train, y_test = preparar_e_dividir_dados(df, coluna_alvo, random_state=seed)

        # 2. Ajuste do Classificador por Árvore de Decisão para Propensão de Campanha
        # max_depth=3 é utilizado para manter o modelo simples e auditável pela equipe de negócios
        modelo_dt = DecisionTreeClassifier(random_state=seed, max_depth=3)
        modelo_dt.fit(X_train, y_train)

        # 3. Predição e Validação das Respostas à Campanha Promocional
        y_pred_dt = modelo_dt.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred_dt)

        # 4. Painel de Métricas de Avaliação de Propensão no Console
        print("\n" + "="*60)
        print("AVALIAÇÃO DO MODELO DE PROPENSÃO A CAMPANHAS (ÁRVORE DE DECISÃO)")
        print("="*60)
        print(f"Acurácia do Classificador de Marketing: {acuracia * 100:.2f}%")
        print("="*60)

        print("\nMatriz de Confusão:")
        print(confusion_matrix(y_test, y_pred_dt))

        print("\nRelatório de Classificação Detalhado (Precisão e Cobertura):")
        print(classification_report(y_test, y_pred_dt))
        print("="*60 + "\n")

        # 5. Exportação da Visualização Gráfica do Fluxograma de Decisões
        exportar_grafico_arvore_propensao(modelo_dt, list(df.columns[:-1]), caminho_img_dt)

    except FileNotFoundError as fnf_err:
        logging.error(fnf_err)
    except Exception as err:
        logging.critical(f"Falha operacional no pipeline de marketing preditivo: {err}", exc_info=True)


if __name__ == "__main__":
    main()