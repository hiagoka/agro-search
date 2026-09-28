"""
AgroSearch - Motor de busca textual com TF-IDF implementado do zero
Startup: AgroTech Solutions

Pipeline: Tokenização -> Normalização -> Stopwords -> Stemming
Índice Invertido em memória + Ranqueamento por TF-IDF (e Similaridade de Cosseno - bônus)

Nenhuma biblioteca de alto nível (scikit-learn, TfidfVectorizer etc.) é utilizada.
Toda a matemática de TF, IDF, TF-IDF e cosseno é implementada manualmente.
"""

import math
import re
import unicodedata
from collections import defaultdict

import pandas as pd
import streamlit as st

# =============================================================================
# BASE DE DOCUMENTOS (hardcoded, conforme especificação)
# =============================================================================

DOCUMENTS = [
    "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico.",
]

# =============================================================================
# FASE 1: PIPELINE DE PRÉ-PROCESSAMENTO
# =============================================================================

# Lista de stopwords em português (curta e suficiente para o domínio do exercício)
STOPWORDS_PT = {
    "a", "o", "as", "os", "de", "do", "da", "dos", "das", "e", "é", "em", "na",
    "no", "nas", "nos", "para", "por", "com", "um", "uma", "uns", "umas", "que",
    "se", "ou", "ao", "aos", "à", "às", "durante", "sua", "seu", "suas", "seus",
}

# Sufixos comuns em português, ordenados do maior para o menor,
# usados por um stemmer simplificado ("do zero") baseado em remoção de sufixo.
SUFFIXES = [
    "amento", "imento", "ização", "izações", "idade", "idades",
    "ções", "ção", "ões", "mente", "adoras", "adora", "adores", "ador",
    "ico", "ica", "icos", "icas", "oso", "osa", "osos", "osas",
    "ante", "antes", "ável", "íveis", "ando", "endo", "indo",
    "s",
]


def remover_acentos(texto: str) -> str:
    """Remove acentuação mantendo apenas caracteres ASCII base."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tokenizar(texto: str) -> list[str]:
    """Normaliza (lowercase + remove acentos) e tokeniza em palavras."""
    texto = texto.lower()
    texto = remover_acentos(texto)
    tokens = re.findall(r"[a-z]+", texto)
    return tokens


def aplicar_stemming(token: str) -> str:
    """Stemmer simplificado por remoção de sufixo (implementado do zero)."""
    for sufixo in SUFFIXES:
        if token.endswith(sufixo) and len(token) - len(sufixo) >= 3:
            return token[: -len(sufixo)]
    return token


def preprocessar(texto: str, usar_stopwords: bool, usar_stemming: bool) -> list[str]:
    """Pipeline completo: tokenização -> normalização -> stopwords -> stemming."""
    tokens = tokenizar(texto)

    if usar_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_PT]

    if usar_stemming:
        tokens = [aplicar_stemming(t) for t in tokens]

    return tokens


# =============================================================================
# FASE 2: ÍNDICE INVERTIDO
# =============================================================================

def construir_indice_invertido(docs_tokens: list[list[str]]) -> dict[str, list[int]]:
    """Constrói o índice invertido Termo -> [IDs de documentos] (1-indexado)."""
    indice = defaultdict(set)

    for doc_id, tokens in enumerate(docs_tokens, start=1):
        for token in tokens:
            indice[token].add(doc_id)

    # Ordena os doc_ids e o vocabulário para exibição consistente
    return {termo: sorted(doc_ids) for termo, doc_ids in sorted(indice.items())}


# =============================================================================
# FASE 3: TF, IDF, TF-IDF E RANQUEAMENTO
# =============================================================================

def calcular_tf(tokens: list[str]) -> dict[str, float]:
    """TF(termo, doc) = (nº de ocorrências do termo) / (nº total de termos no doc)."""
    total = len(tokens)
    if total == 0:
        return {}

    contagem = defaultdict(int)
    for token in tokens:
        contagem[token] += 1

    return {termo: freq / total for termo, freq in contagem.items()}


def calcular_idf(docs_tokens: list[list[str]], vocabulario: set[str]) -> dict[str, float]:
    """IDF(termo) = log10(N / df_termo), onde df_termo é o nº de docs que contêm o termo."""
    n_docs = len(docs_tokens)
    idf = {}

    for termo in vocabulario:
        df = sum(1 for tokens in docs_tokens if termo in tokens)
        idf[termo] = math.log10(n_docs / df) if df > 0 else 0.0

    return idf


def calcular_matriz_tfidf(
    docs_tf: list[dict[str, float]], idf: dict[str, float]
) -> list[dict[str, float]]:
    """Para cada doc, TF-IDF(termo, doc) = TF(termo, doc) * IDF(termo)."""
    return [
        {termo: tf_valor * idf.get(termo, 0.0) for termo, tf_valor in tf_doc.items()}
        for tf_doc in docs_tf
    ]


def similaridade_cosseno(vetor_a: dict[str, float], vetor_b: dict[str, float]) -> float:
    """Cosseno entre dois vetores esparsos representados como dicionários termo -> peso."""
    termos_comuns = set(vetor_a.keys()) & set(vetor_b.keys())
    produto_escalar = sum(vetor_a[t] * vetor_b[t] for t in termos_comuns)

    norma_a = math.sqrt(sum(v ** 2 for v in vetor_a.values()))
    norma_b = math.sqrt(sum(v ** 2 for v in vetor_b.values()))

    if norma_a == 0 or norma_b == 0:
        return 0.0

    return produto_escalar / (norma_a * norma_b)


# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

st.set_page_config(page_title="AgroSearch", page_icon="🌱", layout="wide")

st.title("🌱 AgroSearch")
st.caption("Motor de busca textual sobre manuais técnicos de agricultura — TF-IDF implementado do zero")

# --- Sidebar: configurações do pipeline -------------------------------------
st.sidebar.header("⚙️ Pipeline de Pré-processamento")
usar_stopwords = st.sidebar.checkbox("Remover stopwords", value=True)
usar_stemming = st.sidebar.checkbox("Aplicar stemming", value=True)
usar_cosseno = st.sidebar.checkbox("🏆 Bônus: usar Similaridade de Cosseno", value=False)

# --- Documentos originais -----------------------------------------------------
with st.expander("📄 Base de documentos", expanded=False):
    for i, doc in enumerate(DOCUMENTS, start=1):
        st.write(f"**Doc {i}:** {doc}")

# --- Fase 1: pipeline aplicado -------------------------------------------------
docs_tokens = [preprocessar(doc, usar_stopwords, usar_stemming) for doc in DOCUMENTS]
vocabulario = sorted({token for tokens in docs_tokens for token in tokens})

st.header("1️⃣ Pré-processamento")
col1, col2 = st.columns(2)
with col1:
    st.write("**Tokens por documento**")
    for i, tokens in enumerate(docs_tokens, start=1):
        st.write(f"Doc {i}: `{tokens}`")
with col2:
    st.write(f"**Vocabulário final** ({len(vocabulario)} termos)")
    st.write(vocabulario)

# --- Fase 2: índice invertido ---------------------------------------------------
st.header("2️⃣ Índice Invertido")
indice_invertido = construir_indice_invertido(docs_tokens)
st.json(indice_invertido)

# --- Fase 3: busca e ranqueamento TF-IDF ----------------------------------------
st.header("3️⃣ Busca e Ranqueamento (TF-IDF)")

query = st.text_input("Digite sua consulta:", placeholder="ex: irrigação soja")

if query:
    query_tokens = preprocessar(query, usar_stopwords, usar_stemming)
    st.write(f"Tokens da query: `{query_tokens}`")

    if not query_tokens:
        st.warning("A consulta não gerou nenhum token válido após o pré-processamento.")
    else:
        docs_tf = [calcular_tf(tokens) for tokens in docs_tokens]
        idf = calcular_idf(docs_tokens, set(vocabulario))
        docs_tfidf = calcular_matriz_tfidf(docs_tf, idf)

        query_tf = calcular_tf(query_tokens)
        query_tfidf = {termo: tf_val * idf.get(termo, 0.0) for termo, tf_val in query_tf.items()}

        resultados = []
        for doc_id, doc_tfidf in enumerate(docs_tfidf, start=1):
            score_acumulado = sum(doc_tfidf.get(termo, 0.0) for termo in query_tokens)
            linha = {
                "Doc": f"Doc {doc_id}",
                "Trecho": DOCUMENTS[doc_id - 1][:70] + "...",
                "TF-IDF acumulado": round(score_acumulado, 4),
            }
            if usar_cosseno:
                linha["Similaridade de Cosseno"] = round(
                    similaridade_cosseno(query_tfidf, doc_tfidf), 4
                )
            resultados.append(linha)

        coluna_ordenacao = "Similaridade de Cosseno" if usar_cosseno else "TF-IDF acumulado"
        df_resultados = pd.DataFrame(resultados).sort_values(
            by=coluna_ordenacao, ascending=False
        ).reset_index(drop=True)

        vencedor = df_resultados.iloc[0]
        if vencedor[coluna_ordenacao] > 0:
            st.success(f"🏆 Documento mais relevante: **{vencedor['Doc']}** ({coluna_ordenacao} = {vencedor[coluna_ordenacao]})")
        else:
            st.info("Nenhum documento contém os termos buscados.")

        st.dataframe(
            df_resultados.style.background_gradient(subset=[coluna_ordenacao], cmap="Greens"),
            use_container_width=True,
        )
else:
    st.info("Digite uma consulta acima para ver o ranqueamento dos documentos.")
