def exibir_cotacao_carbono():
    """
    Exibe a cotação do carbono com informações sobre o contrato atual
    """
    st.sidebar.header("💰 Mercado de Carbono e Câmbio")
    
    if not YFINANCE_AVAILABLE:
        st.sidebar.warning("⚠️ **yfinance não instalado**")
        st.sidebar.info("Para cotações em tempo real, execute:")
        st.sidebar.code("pip install yfinance")
    
    # Botão para atualizar cotações
    if st.sidebar.button("🔄 Atualizar Cotações"):
        st.session_state.cotacao_atualizada = True
        # Mostrar mensagem de atualização
        st.sidebar.info("🔄 Atualizando cotações...")
        
        # Forçar um rerun para processar as atualizações
        st.rerun()

    # Obtém informações do contrato atual
    ticker_atual, ano_contrato = obter_ticker_carbono_atual()
    
    if st.session_state.get('cotacao_atualizada', False):
        # Obter cotação do carbono
        preco_carbono, moeda, contrato_info, sucesso_carbono = obter_cotacao_carbono()
        
        # Obter cotação do Euro
        preco_euro, moeda_real, sucesso_euro = obter_cotacao_euro_real()
        
        # Mostrar resultados
        if sucesso_carbono:
            st.sidebar.success(f"**{contrato_info}**")
        else:
            st.sidebar.info(f"**{contrato_info}**")
        
        if sucesso_euro:
            st.sidebar.success(f"**EUR/BRL Atualizado**")
        else:
            st.sidebar.info(f"**EUR/BRL Referência**")
        
        # Atualizar session state
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        st.session_state.ano_contrato = ano_contrato
        
        # Resetar flag para evitar atualizações contínuas
        st.session_state.cotacao_atualizada = False
        
        # Forçar outro rerun para mostrar os valores atualizados
        st.rerun()
    else:
        # Atualizar o ano_contrato se necessário (para caso o ano tenha mudado)
        ticker_atual, ano_contrato_atual = obter_ticker_carbono_atual()
        if st.session_state.ano_contrato != ano_contrato_atual:
            st.session_state.ano_contrato = ano_contrato_atual

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
