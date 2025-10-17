import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from scipy import stats
from scipy.signal import fftconvolve
from joblib import Parallel, delayed
import warnings
from matplotlib.ticker import FuncFormatter
from SALib.sample.sobol import sample
from SALib.analyze.sobol import analyze

# Tentar importar yfinance com fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.warning("⚠️ yfinance não está instalado. Use: `pip install yfinance`")

np.random.seed(50)  # Garante reprodutibilidade

# Configurações iniciais
st.set_page_config(page_title="Simulador de Emissões CO₂eq", layout="wide")
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Título do aplicativo
st.title("Simulador de Emissões de tCO₂eq")
st.markdown("""
Esta ferramenta calcula as emissões de gases de efeito estufa para dois contextos de gestão de resíduos,
aterro sanitário vs. vermicompostagem (Contexto: Proposta da Tese) e aterro sanitário vs. compostagem (Contexto: UNFCCC).
""")

# =============================================================================
# FUNÇÕES DE COTAÇÃO AUTOMÁTICA DO CARBONO E CÂMBIO
# =============================================================================

def obter_ticker_carbono_atual():
    """
    Determina automaticamente o ticker do contrato futuro de carbono mais relevante
    """
    ano_atual = datetime.now().year
    mes_atual = datetime.now().month
    
    # Lógica: a partir de setembro, começa a migrar para o próximo ano
    if mes_atual >= 9:
        ano_contrato = ano_atual + 1
    else:
        ano_contrato = ano_atual
    
    # Formata o ano para 2 dígitos (25, 26, etc.)
    ano_2_digitos = str(ano_contrato)[-2:]
    
    ticker = f'CO2Z{ano_2_digitos}.NYB'
    return ticker, ano_contrato

def obter_cotacao_carbono():
    """
    Obtém a cotação em tempo real do contrato futuro de carbono atual
    """
    if not YFINANCE_AVAILABLE:
        ticker_atual, ano_contrato = obter_ticker_carbono_atual()
        return 85.50, "€", f"EUA Carbon Dec {ano_contrato} (yfinance não disponível)", False
    
    try:
        # Obtém o ticker atual automaticamente
        ticker_atual, ano_contrato = obter_ticker_carbono_atual()
        ano_2_digitos = str(ano_contrato)[-2:]
        
        simbolos_tentativas = [
            ticker_atual,                    # Contrato atual (ex: CO2Z25.NYB)
            f'CFIZ{ano_2_digitos}.NYB',     # Alternativa com mesmo ano
            'CARBON-FUTURE',                # Genérico
        ]
        
        cotacao = None
        simbolo_usado = None
        
        for simbolo in simbolos_tentativas:
            try:
                ticker = yf.Ticker(simbolo)
                hist = ticker.history(period='1d')
                
                if not hist.empty and not pd.isna(hist['Close'].iloc[-1]):
                    cotacao = hist['Close'].iloc[-1]
                    simbolo_usado = simbolo
                    break
                    
            except Exception as e:
                continue
        
        if cotacao is None:
            # Fallback para dados de exemplo
            return 85.50, "€", f"EUA Carbon Dec {ano_contrato} (Referência)", False
        
        return cotacao, "€", f"EUA Carbon Futures Dec {ano_contrato}", True
        
    except Exception as e:
        ticker_atual, ano_contrato = obter_ticker_carbono_atual()
        return 85.50, "€", f"EUA Carbon Dec {ano_contrato} (Erro: {str(e)})", False

def obter_cotacao_euro_real():
    """
    Obtém a cotação em tempo real do Euro em relação ao Real Brasileiro
    """
    if not YFINANCE_AVAILABLE:
        return 5.50, "R$", False
    
    try:
        # Ticker para EUR/BRL (Euro para Real Brasileiro)
        ticker = yf.Ticker("EURBRL=X")
        hist = ticker.history(period='1d')
        
        if not hist.empty and not pd.isna(hist['Close'].iloc[-1]):
            cotacao = hist['Close'].iloc[-1]
            return cotacao, "R$", True
        else:
            # Fallback para valor de referência
            return 5.50, "R$", False
            
    except Exception as e:
        return 5.50, "R$", False

def calcular_valor_creditos(emissoes_evitadas_tco2eq, preco_carbono_por_tonelada, moeda, taxa_cambio=1):
    """
    Calcula o valor financeiro das emissões evitadas baseado no preço do carbono
    """
    valor_total = emissoes_evitadas_tco2eq * preco_carbono_por_tonelada * taxa_cambio
    return valor_total

def exibir_cotacao_carbono():
    """
    Exibe a cotação do carbono com informações sobre o contrato atual
    """
    st.sidebar.header("💰 Mercado de Carbono e Câmbio")
    
    if not YFINANCE_AVAILABLE:
        st.sidebar.warning("⚠️ **yfinance não instalado**")
        st.sidebar.info("Para cotações em tempo real, execute:")
        st.sidebar.code("pip install yfinance")
    
    if st.sidebar.button("🔄 Atualizar Cotações"):
        st.session_state.cotacao_atualizada = True

    # Obtém informações do contrato atual
    ticker_atual, ano_contrato = obter_ticker_carbono_atual()
    
    if st.session_state.get('cotacao_atualizada', False):
        with st.sidebar.spinner('Obtendo cotações...'):
            # Obter cotação do carbono
            preco_carbono, moeda, contrato_info, sucesso_carbono = obter_cotacao_carbono()
            
            # Obter cotação do Euro
            preco_euro, moeda_real, sucesso_euro = obter_cotacao_euro_real()
            
            if sucesso_carbono:
                st.sidebar.success(f"**{contrato_info}**")
            else:
                st.sidebar.info(f"**{contrato_info}**")
            
            if sucesso_euro:
                st.sidebar.success(f"**EUR/BRL Atualizado**")
            else:
                st.sidebar.info(f"**EUR/BRL Referência**")
            
            st.session_state.preco_carbono = preco_carbono
            st.session_state.moeda_carbono = moeda
            st.session_state.contrato_info = contrato_info
            st.session_state.taxa_cambio = preco_euro
            st.session_state.moeda_real = moeda_real
            st.session_state.ano_contrato = ano_contrato  # Armazena o ano atual
    else:
        # Valores padrão iniciais
        if 'preco_carbono' not in st.session_state:
            st.session_state.preco_carbono = 85.50
            st.session_state.moeda_carbono = "€"
            st.session_state.contrato_info = f"EUA Carbon Dec {ano_contrato}"
            st.session_state.taxa_cambio = 5.50
            st.session_state.moeda_real = "R$"
            st.session_state.ano_contrato = ano_contrato  # Armazena o ano atual

    # Exibe cotação atual do carbono
    st.sidebar.metric(
        label=f"Carbon Dec {st.session_state.ano_contrato} (tCO₂eq)",
        value=f"{st.session_state.moeda_carbono} {st.session_state.preco_carbono:.2f}",
        help=f"Contrato futuro com vencimento Dezembro {st.session_state.ano_contrato}"
    )
    
    # Exibe cotação atual do Euro
    st.sidebar.metric(
        label="Euro (EUR/BRL)",
        value=f"{st.session_state.moeda_real} {st.session_state.taxa_cambio:.2f}",
        help="Cotação do Euro em Reais Brasileiros"
    )
    
    # Calcular preço do carbono em Reais
    preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
    
    st.sidebar.metric(
        label=f"Carbon Dec {st.session_state.ano_contrato} (R$/tCO₂eq)",
        value=f"R$ {preco_carbono_reais:.2f}",
        help="Preço do carbono convertido para Reais Brasileiros"
    )
    
    # Informações adicionais
    with st.sidebar.expander("📅 Sobre os Vencimentos e Câmbio"):
        st.markdown(f"""
        **Contrato Atual:** Dec {st.session_state.ano_contrato}
        **Ticker:** `{ticker_atual}`
        
        **Câmbio Atual:**
        - 1 Euro = R$ {st.session_state.taxa_cambio:.2f}
        - Carbon em Reais: R$ {preco_carbono_reais:.2f}/tCO₂eq
        
        **Ciclo dos Contratos:**
        - Dez 2024 → CO2Z24.NYB
        - Dez 2025 → CO2Z25.NYB  
        - Dez 2026 → CO2Z26.NYB
        - Dez 2027 → CO2Z27.NYB
        
        **Migração Automática:**
        - A partir de Setembro: prepara para próximo ano
        - O app ajusta automaticamente
        - Sem necessidade de atualização manual
        """)

# =============================================================================
# FUNÇÕES ORIGINAIS DO SEU SCRIPT (MANTIDAS)
# =============================================================================

# [TODO: COLE AQUI TODAS AS SUAS FUNÇÕES ORIGINAIS, DESDE formatar_br ATÉ executar_simulacao_unfccc]

# =============================================================================
# EXECUÇÃO DA SIMULAÇÃO - PARTE MODIFICADA
# =============================================================================

# Executar simulação quando solicitado
if st.session_state.get('run_simulation', False):
    with st.spinner('Executando simulação...'):
        # [TODO: COLE AQUI TODO O SEU CÓDIGO DE SIMULAÇÃO ORIGINAL]
        
        # =============================================================================
        # EXIBIÇÃO DOS RESULTADOS COM COTAÇÃO DO CARBONO E REAL - PARTE CORRIGIDA
        # =============================================================================

        # Exibir resultados
        st.header("Resultados da Simulação")
        
        # Obter valores totais
        total_evitado_tese = df['Reducao_tCO2eq_acum'].iloc[-1]
        total_evitado_unfccc = df_comp_anual_revisado['Cumulative reduction (t CO₂eq)'].iloc[-1]
        
        # Obter preço do carbono e taxa de câmbio da session state
        preco_carbono = st.session_state.preco_carbono
        moeda = st.session_state.moeda_carbono
        taxa_cambio = st.session_state.taxa_cambio
        ano_contrato = st.session_state.ano_contrato  # Usa o ano armazenado
        
        # Calcular valores financeiros em Euros
        valor_tese_eur = calcular_valor_creditos(total_evitado_tese, preco_carbono, moeda)
        valor_unfccc_eur = calcular_valor_creditos(total_evitado_unfccc, preco_carbono, moeda)
        
        # Calcular valores financeiros em Reais
        valor_tese_brl = calcular_valor_creditos(total_evitado_tese, preco_carbono, "R$", taxa_cambio)
        valor_unfccc_brl = calcular_valor_creditos(total_evitado_unfccc, preco_carbono, "R$", taxa_cambio)
        
        # NOVA SEÇÃO: VALOR FINANCEIRO DAS EMISSÕES EVITADAS
        st.subheader("💰 Valor Financeiro das Emissões Evitadas")
        
        if not YFINANCE_AVAILABLE:
            st.warning("⚠️ **Cotações em modo offline** - Instale yfinance para valores em tempo real")
        
        # Primeira linha: Euros
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                f"Preço Carbon Dec {ano_contrato} (Euro)", 
                f"{moeda} {preco_carbono:.2f}/tCO₂eq",
                help=f"Cotação do contrato futuro para Dezembro {ano_contrato}"
            )
        with col2:
            st.metric(
                "Valor Tese (Euro)", 
                f"{moeda} {formatar_br(valor_tese_eur)}",
                help=f"Baseado em {formatar_br(total_evitado_tese)} tCO₂eq evitadas"
            )
        with col3:
            st.metric(
                "Valor UNFCCC (Euro)", 
                f"{moeda} {formatar_br(valor_unfccc_eur)}",
                help=f"Baseado em {formatar_br(total_evitado_unfccc)} tCO₂eq evitadas"
            )
        
        # Segunda linha: Reais
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                f"Preço Carbon Dec {ano_contrato} (Real)", 
                f"R$ {formatar_br(preco_carbono * taxa_cambio)}/tCO₂eq",
                help="Preço do carbono convertido para Reais"
            )
        with col2:
            st.metric(
                "Valor Tese (Real)", 
                f"R$ {formatar_br(valor_tese_brl)}",
                help=f"Baseado em {formatar_br(total_evitado_tese)} tCO₂eq evitadas"
            )
        with col3:
            st.metric(
                "Valor UNFCCC (Real)", 
                f"R$ {formatar_br(valor_unfccc_brl)}",
                help=f"Baseado em {formatar_br(total_evitado_unfccc)} tCO₂eq evitadas"
            )
        
        # Explicação sobre compra e venda
        with st.expander("💡 Como funciona a comercialização no mercado de carbono?"):
            st.markdown(f"""
            **Para o Carbon Dec {ano_contrato}:**
            - **Preço em Euro:** {moeda} {preco_carbono:.2f}/tCO₂eq
            - **Preço em Real:** R$ {formatar_br(preco_carbono * taxa_cambio)}/tCO₂eq
            - **Taxa de câmbio:** 1 Euro = R$ {taxa_cambio:.2f}
            
            **📈 Comprar créditos (compensação):**
            - Custo em Euro: **{moeda} {formatar_br(valor_tese_eur)}**
            - Custo em Real: **R$ {formatar_br(valor_tese_brl)}**
            
            **📉 Vender créditos (comercialização):**  
            - Receita em Euro: **{moeda} {formatar_br(valor_tese_eur)}**
            - Receita em Real: **R$ {formatar_br(valor_tese_brl)}**
            
            **Contrato Carbon Dec {ano_contrato}:**
            - Cada contrato = 1.000 tCO₂eq
            - Vencimento: Dezembro {ano_contrato}
            - Mercado: ICE Exchange
            - Moeda original: Euros (€)
            - Ticker no Yahoo Finance: `CO2Z{str(ano_contrato)[-2:]}.NYB`
            """)
        
        # [TODO: O RESTO DO SEU CÓDIGO ORIGINAL PERMANECE AQUI]

else:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Executar Simulação' para ver os resultados.")

# Rodapé
st.markdown("---")
st.markdown("""
**Referências por Cenário:**

**Cenário de Baseline (Aterro Sanitário):**
- IPCC (2006). Guidelines for National Greenhouse Gas Inventories.
- UNFCCC (2016). Tool to determine methane emissions from solid waste disposal sites.
- Wang et al. (2017). Nitrous oxide emissions from landfills.
- Feng et al. (2020). Emissions from pre-disposal organic waste.

**Proposta da Tese (Vermicompostagem):**
- Yang et al. (2017). Greenhouse gas emissions from vermicomposting.

**Cenário UNFCCC (Compostagem):**
- UNFCCC (2012). AMS-III.F - Methodology for compostage.
- Yang et al. (2017). Greenhouse gas emissions from thermophilic composting.
""")
