import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(page_title="Análise: Desigualdade vs Criminalidade", layout="wide")

# ============================================================================
# SEÇÃO 1: CARREGAMENTO E PROCESSAMENTO DE DADOS
# ============================================================================

@st.cache_data
def load_and_process_data():
    """Carrega dados integrados (crime + ipeadata + SUS + Base_MUNIC 2021)"""
    
    # Caminho base
    base_path = ""
    
    try:
        # Carregar dataset integrado final
        data = pd.read_csv(f"dados_integrados_ride_final.csv")
        
        # Limpar dados
        data = data.dropna(subset=['crimes_total', 'desigualdade'])
        data = data[data['crimes_total'] > 0].copy()
        
        if len(data) == 0:
            st.error("Nenhum dado valido apos carregamento")
            return None
        
        return data
    
    except FileNotFoundError:
        st.error(f"dados_integrados_ride_final.csv")
        return None
    
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Carregar dados
data = load_and_process_data()

if data is None or len(data) == 0:
    st.error("Não foi possível carregar os dados. Verifique os arquivos CSV.")
    st.stop()

# ============================================================================
# SEÇÃO 2: NAVEGAÇÃO E PÁGINAS
# ============================================================================

st.sidebar.title("Analise: Desigualdade vs Criminalidade")
page = st.sidebar.radio(
    "Selecione a pagina:",
    ["Resumo Executivo", 
     "Analise Descritiva", 
     "Correlacoes e Regressoes",
     "Validacao Cruzada",
     "Infraestrutura & Capacidade",
     "Conclusoes"]
)

# ============================================================================
# PÁGINA 1: RESUMO EXECUTIVO
# ============================================================================

if page == "Resumo Executivo":
    st.title("Resumo Executivo")
    st.markdown("## Regiões com altos índices de desigualdade social apresentam taxas de criminalidade mais elevadas?")
    
    # Calcular métricas
    correlation = data['desigualdade'].corr(data['crimes_total'])
    avg_inequality = data['desigualdade'].mean()
    total_crimes = data['crimes_total'].sum()
    avg_crimes_per_municipality = data.groupby('municipio')['crimes_total'].sum().mean()
    
    # Regressão linear para p-valor
    X = data[['desigualdade']]
    y = data['crimes_total']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    p_value = model.pvalues['desigualdade']
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Índice de Desigualdade (Média)", f"{avg_inequality:.3f}")
    with col2:
        st.metric("Crimes Totais (2022-2023)", f"{int(total_crimes)}")
    with col3:
        st.metric("Crimes por Município (Média)", f"{int(avg_crimes_per_municipality)}")
    with col4:
        st.metric("Correlação Pearson", f"{correlation:.3f}")
    
    # Interpretação da correlação
    st.markdown("---")
    
    if abs(correlation) < 0.3:
        corr_strength = "fraca"
    elif abs(correlation) < 0.7:
        corr_strength = "moderada"
    else:
        corr_strength = "forte"
    
    if correlation > 0:
        corr_direction = "positiva"
        direction_text = "aumentam"
    else:
        corr_direction = "negativa"
        direction_text = "diminuem"
    
    significance = "✅ ESTATISTICAMENTE SIGNIFICATIVA" if p_value < 0.05 else "⚠️ NÃO SIGNIFICATIVA"
    
    if p_value < 0.05 and correlation > 0:
        hypothesis_status = "✅ HIPÓTESE CONFIRMADA"
        color = "green"
    elif p_value < 0.05 and correlation < 0:
        hypothesis_status = "❌ HIPÓTESE REFUTADA (relação negativa)"
        color = "red"
    else:
        hypothesis_status = "⚠️ HIPÓTESE INCONCLUSIVA"
        color = "orange"
    
    st.markdown(f"""
    ### Interpretação dos Resultados
    
    A análise revela uma correlação **{corr_direction} e {corr_strength}** (r = {correlation:.3f}) entre 
    desigualdade social e criminalidade no período 2022-2023.
    
    **P-valor: {p_value:.4f}** - {significance}
    
    ---
    
    #### {hypothesis_status}
    
    - Quando o índice de desigualdade aumenta, os crimes **{direction_text}** {f"em média {correlation:.3f} unidades por 0.1 de aumento na desigualdade" if abs(correlation) > 0.1 else "ligeiramente"}
    - A relação é **{"estatisticamente significativa (α=0.05)" if p_value < 0.05 else "não significativa ao nível de 5%"}**
    - R² do modelo: {model.rsquared:.3f} (o índice de desigualdade explica {model.rsquared*100:.1f}% da variação em crimes)
    """)
    
    # Gráfico scatter com regressão
    st.markdown("---")
    st.subheader("Visualização: Desigualdade vs Criminalidade")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(data['desigualdade'], data['crimes_total'], alpha=0.6, s=100, color='steelblue')
    
    # Linha de regressão
    z = np.polyfit(data['desigualdade'], data['crimes_total'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(data['desigualdade'].min(), data['desigualdade'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Regressão Linear (R²={model.rsquared:.3f})')
    
    ax.set_xlabel('Índice de Desigualdade Social', fontsize=12, fontweight='bold')
    ax.set_ylabel('Número Total de Crimes', fontsize=12, fontweight='bold')
    ax.set_title('Relação entre Desigualdade Social e Criminalidade\n(RIDE 2022-2023)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# PÁGINA 2: ANÁLISE DESCRITIVA
# ============================================================================

elif page == "Analise Descritiva":
    st.title("Análise Descritiva")
    
    # Estatísticas por ano
    st.subheader("Estatísticas Básicas por Ano")
    
    stats_by_year = data.groupby('ano').agg({
        'desigualdade': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'crimes_total': ['sum', 'mean', 'median', 'std', 'min', 'max']
    }).round(3)
    
    st.dataframe(stats_by_year)
    
    # Comentário interpretativo
    crime_2022 = data[data['ano'] == 2022]['crimes_total'].sum()
    crime_2023 = data[data['ano'] == 2023]['crimes_total'].sum()
    crime_change = ((crime_2023 - crime_2022) / crime_2022 * 100) if crime_2022 > 0 else 0
    
    desig_2022 = data[data['ano'] == 2022]['desigualdade'].mean()
    desig_2023 = data[data['ano'] == 2023]['desigualdade'].mean()
    desig_change = ((desig_2023 - desig_2022) / desig_2022 * 100) if desig_2022 > 0 else 0
    
    st.markdown("---")
    st.markdown(f"""
    ### Interpretação dos Padrões Observados
    
    **Criminalidade:**
    - Total de crimes em 2022: {int(crime_2022)}
    - Total de crimes em 2023: {int(crime_2023)}
    - Variação: {crime_change:+.1f}%
    - **Observação:** A criminalidade {"aumentou" if crime_change > 0 else "diminuiu"} de 2022 para 2023 na região RIDE
    
    **Desigualdade Social:**
    - Índice médio 2022: {desig_2022:.3f}
    - Índice médio 2023: {desig_2023:.3f}
    - Variação: {desig_change:+.1f}%
    - **Observação:** O índice de desigualdade {"aumentou" if desig_change > 0 else "diminuiu"} no período
    """)
    
    # Visualizações
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição do Índice de Desigualdade")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(data['desigualdade'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Índice de Desigualdade', fontweight='bold')
        ax.set_ylabel('Frequência', fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Distribuição de Crimes Totais")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(data['crimes_total'], bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Total de Crimes', fontweight='bold')
        ax.set_ylabel('Frequência', fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Box plots por ano
    st.subheader("Comparação de Distribuições por Ano")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        data.boxplot(column='desigualdade', by='ano', ax=ax)
        ax.set_title('Desigualdade por Ano', fontweight='bold')
        ax.set_xlabel('Ano')
        ax.set_ylabel('Índice de Desigualdade')
        plt.suptitle('')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        data.boxplot(column='crimes_total', by='ano', ax=ax)
        ax.set_title('Crimes por Ano', fontweight='bold')
        ax.set_xlabel('Ano')
        ax.set_ylabel('Total de Crimes')
        plt.suptitle('')
        st.pyplot(fig)

# ============================================================================
# PÁGINA 3: CORRELAÇÕES E REGRESSÕES
# ============================================================================

elif page == "Correlacoes e Regressoes":
    st.title("Correlações e Regressões")
    
    # Preparar dados
    X = data[['desigualdade']]
    y = data['crimes_total']
    
    # Regressão com statsmodels
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()
    
    # Mostrar resumo
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Resultados da Regressão Linear")
        st.text(model.summary().as_text())
    
    with col2:
        st.subheader("Interpretação Estatística")
        
        coef = model.params['desigualdade']
        pval = model.pvalues['desigualdade']
        r2 = model.rsquared
        adj_r2 = model.rsquared_adj
        f_stat = model.fvalue
        f_pval = model.f_pvalue
        
        st.markdown(f"""
        **Coeficiente de Desigualdade:** {coef:.4f}
        - P-valor: {pval:.6f}
        - Significância: {"✅ SIM (p < 0.05)" if pval < 0.05 else "❌ NÃO (p ≥ 0.05)"}
        
        **Qualidade do Modelo:**
        - R²: {r2:.4f}
        - R² ajustado: {adj_r2:.4f}
        - F-estatística: {f_stat:.4f}
        - P-valor (F): {f_pval:.6f}
        
        **Tamanho da Amostra:** {len(data)} observações
        """)
    
    # Interpretação
    st.markdown("---")
    st.markdown("### Interpretação dos Resultados")
    
    if pval < 0.05:
        st.success(f"""
        ✅ **RELAÇÃO ESTATISTICAMENTE SIGNIFICATIVA**
        
        Para cada aumento de 0.1 no índice de desigualdade, o número de crimes 
        aumenta em média **{coef * 0.1:.2f}** (p = {pval:.6f}).
        
        O modelo explica **{r2*100:.1f}%** da variação nos crimes.
        """)
    else:
        st.warning(f"""
        ⚠️ **RELAÇÃO NÃO SIGNIFICATIVA**
        
        Não há evidência estatística significativa de que a desigualdade 
        afete a criminalidade nesta amostra (p = {pval:.6f} > 0.05).
        """)
    
    # Scatter com regressão
    st.subheader("Visualização: Regressão Linear")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data['desigualdade'], data['crimes_total'], alpha=0.6, s=100, color='steelblue')
    
    z = np.polyfit(data['desigualdade'], data['crimes_total'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(data['desigualdade'].min(), data['desigualdade'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2.5, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
    
    ax.set_xlabel('Índice de Desigualdade', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total de Crimes', fontsize=12, fontweight='bold')
    ax.set_title(f'Regressão Linear: Desigualdade vs Criminalidade\n(R² = {r2:.3f}, p = {pval:.6f})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# PÁGINA 4: VALIDAÇÃO CRUZADA
# ============================================================================

elif page == "Validacao Cruzada":
    st.title("Validação Cruzada e Avaliação de Modelos")
    
    X = data[['desigualdade']]
    y = data['crimes_total']
    
    st.markdown("""
    Implementamos 5 métodos de validação cruzada para avaliar a robustez do modelo:
    1. **Holdout 70-30**: 70% treino, 30% teste
    2. **Holdout 80-20**: 80% treino, 20% teste
    3. **K-Fold (k=5)**: Divisão em 5 folds
    4. **K-Fold (k=10)**: Divisão em 10 folds
    5. **LOOCV**: Leave-One-Out Cross-Validation
    """)
    
    # Função auxiliar
    def train_and_evaluate(X_train, y_train, X_test, y_test):
        model = LinearRegression().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        return train_mse, test_mse, train_r2, test_r2
    
    results = {}
    
    # 1. Holdout 70-30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    train_mse, test_mse, train_r2, test_r2 = train_and_evaluate(X_train, y_train, X_test, y_test)
    results['Holdout 70-30'] = {
        'Train MSE': train_mse, 'Test MSE': test_mse,
        'Train R²': train_r2, 'Test R²': test_r2
    }
    
    # 2. Holdout 80-20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    train_mse, test_mse, train_r2, test_r2 = train_and_evaluate(X_train, y_train, X_test, y_test)
    results['Holdout 80-20'] = {
        'Train MSE': train_mse, 'Test MSE': test_mse,
        'Train R²': train_r2, 'Test R²': test_r2
    }
    
    # 3. K-Fold (k=5)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mse_scores_5 = []
    cv_r2_scores_5 = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        cv_mse_scores_5.append(mean_squared_error(y_test, y_pred))
        cv_r2_scores_5.append(r2_score(y_test, y_pred))
    
    results['K-Fold (k=5)'] = {
        'Mean MSE': np.mean(cv_mse_scores_5),
        'Std MSE': np.std(cv_mse_scores_5),
        'Mean R²': np.mean(cv_r2_scores_5),
        'Std R²': np.std(cv_r2_scores_5)
    }
    
    # 4. K-Fold (k=10)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_mse_scores_10 = []
    cv_r2_scores_10 = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        cv_mse_scores_10.append(mean_squared_error(y_test, y_pred))
        cv_r2_scores_10.append(r2_score(y_test, y_pred))
    
    results['K-Fold (k=10)'] = {
        'Mean MSE': np.mean(cv_mse_scores_10),
        'Std MSE': np.std(cv_mse_scores_10),
        'Mean R²': np.mean(cv_r2_scores_10),
        'Std R²': np.std(cv_r2_scores_10)
    }
    
    # 5. LOOCV
    loo = LeaveOneOut()
    loocv_mse_scores = []
    loocv_r2_scores = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        loocv_mse_scores.append(mean_squared_error(y_test, y_pred))
        loocv_r2_scores.append(r2_score(y_test, y_pred))
    
    results['LOOCV'] = {
        'Mean MSE': np.mean(loocv_mse_scores),
        'Std MSE': np.std(loocv_mse_scores),
        'Mean R²': np.mean(loocv_r2_scores),
        'Std R²': np.std(loocv_r2_scores)
    }
    
    # Exibir resultados
    st.subheader("Resultados Comparativos")
    
    # Converter para DataFrame para melhor visualização
    results_df = pd.DataFrame(results).T
    st.dataframe(results_df.round(4))
    
    # Interpretação
    st.markdown("---")
    st.markdown("### Análise dos Métodos de Validação")
    
    holdout_70_overfitting = results['Holdout 70-30']['Test MSE'] - results['Holdout 70-30']['Train MSE']
    holdout_80_overfitting = results['Holdout 80-20']['Test MSE'] - results['Holdout 80-20']['Train MSE']
    
    st.markdown(f"""
    **Análise de Overfitting:**
    - Holdout 70-30: Diferença MSE = {holdout_70_overfitting:.2f}
    - Holdout 80-20: Diferença MSE = {holdout_80_overfitting:.2f}
    
    {"✅ Não há evidência significativa de overfitting" if holdout_70_overfitting < 1000 else "⚠️ Possível overfitting detectado"}
    
    **Melhor Método de Validação:**
    - K-Fold (k=5) é recomendado por ter equilíbrio entre viés e variância
    - K-Fold (k=10) é mais robusto mas computacionalmente mais custoso
    - Para amostras pequenas (n=70), LOOCV é aproximadamente não-viesado
    
    **Estabilidade do Modelo:**
    - Desvio padrão K-Fold (k=5): {results['K-Fold (k=5)']['Std MSE']:.2f}
    - Desvio padrão K-Fold (k=10): {results['K-Fold (k=10)']['Std MSE']:.2f}
    
    {"✅ Modelo apresenta generalização estável" if np.mean([results['K-Fold (k=5)']['Std MSE'], results['K-Fold (k=10)']['Std MSE']]) < 500 else "⚠️ Alta variabilidade em predições"}
    """)
    
    # Gráfico comparativo
    st.subheader("Visualização Comparativa")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE comparativo
    methods = ['Holdout\n70-30', 'Holdout\n80-20', 'K-Fold\nk=5', 'K-Fold\nk=10', 'LOOCV']
    holdout_70_test = [results['Holdout 70-30']['Test MSE']]
    holdout_80_test = [results['Holdout 80-20']['Test MSE']]
    kfold_5_mean = [results['K-Fold (k=5)']['Mean MSE']]
    kfold_10_mean = [results['K-Fold (k=10)']['Mean MSE']]
    loocv_mean = [results['LOOCV']['Mean MSE']]
    
    mse_values = [results['Holdout 70-30']['Test MSE'], 
                  results['Holdout 80-20']['Test MSE'],
                  results['K-Fold (k=5)']['Mean MSE'],
                  results['K-Fold (k=10)']['Mean MSE'],
                  results['LOOCV']['Mean MSE']]
    
    axes[0].bar(methods, mse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    axes[0].set_ylabel('Mean Squared Error (MSE)', fontweight='bold')
    axes[0].set_title('Comparação de MSE entre Métodos', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # R² comparativo
    r2_values = [results['Holdout 70-30']['Test R²'],
                 results['Holdout 80-20']['Test R²'],
                 results['K-Fold (k=5)']['Mean R²'],
                 results['K-Fold (k=10)']['Mean R²'],
                 results['LOOCV']['Mean R²']]
    
    axes[1].bar(methods, r2_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    axes[1].set_ylabel('R² Score', fontweight='bold')
    axes[1].set_title('Comparação de R² entre Métodos', fontweight='bold')
    axes[1].set_ylim([min(r2_values) - 0.1, max(r2_values) + 0.1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# PÁGINA 5: INFRAESTRUTURA & CAPACIDADE INSTITUCIONAL
# ============================================================================

elif page == "Infraestrutura & Capacidade":
    st.title("Infraestrutura & Capacidade Institucional (Base_MUNIC 2021)")
    
    st.markdown("""
    Esta pagina analisa como indicadores de infraestrutura e capacidade institucional
    (medidos em 2021 pela Pesquisa de Informacoes Basicas Municipais - MUNIC)
    se relacionam com as taxas de criminalidade observadas em 2022-2023.
    
    **Nota Temporal**: Os dados de infraestrutura sao de 2021 (baseline), enquanto
    os crimes sao de 2022-2023. Assume-se que a infraestrutura municipal muda lentamente.
    """)
    
    # Verificar se as colunas de indicadores existem
    indicador_cols = ['saude_score', 'educacao_score', 'rh_score', 'legislacao_score']
    data_with_indicators = data.dropna(subset=indicador_cols)
    
    if len(data_with_indicators) == 0:
        st.error("Dados de indicadores nao disponiveis")
    else:
        st.markdown("---")
        st.subheader("Indicadores de Infraestrutura (2021)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Saude (media)", f"{data_with_indicators['saude_score'].mean():.3f}")
        with col2:
            st.metric("Educacao (media)", f"{data_with_indicators['educacao_score'].mean():.3f}")
        with col3:
            st.metric("RH/Capacidade (media)", f"{data_with_indicators['rh_score'].mean():.3f}")
        with col4:
            st.metric("Legislacao/Planejamento (media)", f"{data_with_indicators['legislacao_score'].mean():.3f}")
        
        st.markdown("---")
        
        # Correlacoes entre indicadores e crimes
        st.subheader("Correlacoes: Indicadores vs Criminalidade")
        
        correlacoes = {}
        for col in indicador_cols:
            corr = data_with_indicators['crimes_total'].corr(data_with_indicators[col])
            correlacoes[col.replace('_score', '')] = corr
        
        corr_df = pd.DataFrame.from_dict(correlacoes, orient='index', columns=['Correlacao com Crimes'])
        corr_df = corr_df.sort_values('Correlacao com Crimes', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['green' if x > 0 else 'red' for x in corr_df['Correlacao com Crimes']]
        ax.barh(corr_df.index, corr_df['Correlacao com Crimes'], color=colors, alpha=0.7)
        ax.set_xlabel('Correlacao de Pearson', fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title('Relacao entre Indicadores de Infraestrutura e Crimes', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig)
        
        st.dataframe(corr_df.round(4))
        
        st.markdown("---")
        
        # Scatter plots: Crime vs cada indicador
        st.subheader("Visualizacoes: Crime vs Indicadores")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(data_with_indicators['saude_score'], data_with_indicators['crimes_total'], 
                      alpha=0.6, s=100, color='steelblue')
            z = np.polyfit(data_with_indicators['saude_score'], data_with_indicators['crimes_total'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data_with_indicators['saude_score'].min(), 
                                data_with_indicators['saude_score'].max(), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=2)
            ax.set_xlabel('Score de Infraestrutura de Saude', fontweight='bold')
            ax.set_ylabel('Total de Crimes', fontweight='bold')
            ax.set_title('Saude vs Criminalidade', fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(data_with_indicators['educacao_score'], data_with_indicators['crimes_total'], 
                      alpha=0.6, s=100, color='darkorange')
            z = np.polyfit(data_with_indicators['educacao_score'], data_with_indicators['crimes_total'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data_with_indicators['educacao_score'].min(), 
                                data_with_indicators['educacao_score'].max(), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=2)
            ax.set_xlabel('Score de Infraestrutura de Educacao', fontweight='bold')
            ax.set_ylabel('Total de Crimes', fontweight='bold')
            ax.set_title('Educacao vs Criminalidade', fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Tabela de municipios
        st.markdown("---")
        st.subheader("Comparacao de Municipios: Indicadores e Crimes (2022-2023)")
        
        munic_stats = data_with_indicators.groupby('municipio').agg({
            'crimes_total': 'sum',
            'desigualdade': 'mean',
            'saude_score': 'first',
            'educacao_score': 'first',
            'legislacao_score': 'first'
        }).round(3).sort_values('crimes_total', ascending=False)
        
        munic_stats.columns = ['Total Crimes', 'Desigualdade (media)', 
                               'Saude Score', 'Educacao Score', 'Legislacao Score']
        
        st.dataframe(munic_stats)

# ============================================================================
# PÁGINA 6: CONCLUSÕES
# ============================================================================

elif page == "Conclusoes":
    st.title("Conclusoes e Limitacoes")
    
    # Análise final
    correlation = data['desigualdade'].corr(data['crimes_total'])
    X = data[['desigualdade']]
    y = data['crimes_total']
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()
    p_value = model.pvalues['desigualdade']
    
    # Conclusão principal
    st.markdown("## Resposta a Pergunta de Pesquisa")
    
    if p_value < 0.05 and correlation > 0:
        st.success(f"""
        ### HIPOTESE CONFIRMADA
        
        **Com base na analise de dados de 2022-2023 na regiao RIDE:**
        
        Existe evidencia estatistica significativa de que **regioes com altos indices de 
        desigualdade social apresentam taxas de criminalidade mais elevadas**.
        
        - Correlacao: {correlation:.3f} (positiva e {("forte" if abs(correlation) > 0.7 else ("moderada" if abs(correlation) > 0.3 else "fraca"))})
        - P-valor: {p_value:.6f} (significativo ao nivel α=0.05)
        - Para cada aumento de 0.1 no indice de desigualdade, crimes aumentam em media {model.params['desigualdade'] * 0.1:.2f}
        - O modelo explica {model.rsquared*100:.1f}% da variacao em crimes
        """)
    elif p_value < 0.05 and correlation < 0:
        st.error(f"""
        ### HIPOTESE REFUTADA
        
        A analise revela uma relacao **inversa** e estatisticamente significativa: 
        regioes com **maior desigualdade apresentam MENOR criminalidade** nesta amostra.
        
        - Correlacao: {correlation:.3f} (negativa)
        - P-valor: {p_value:.6f}
        - Este resultado contradiz a literatura esperada e pode indicar confundidores nao incluidos
        """)
    else:
        st.warning(f"""
        ### HIPOTESE INCONCLUSIVA
        
        Nao ha evidencia estatistica suficiente para confirmar a relacao entre 
        desigualdade social e criminalidade nesta amostra.
        
        - Correlacao: {correlation:.3f}
        - P-valor: {p_value:.6f} (nao significativo ao nivel α=0.05)
        - Possiveis causas: amostra pequena, periodo curto, falta de confundidores
        """)
    
    # Limitações
    st.markdown("---")
    st.markdown("## Limitacoes da Analise")
    
    st.markdown(f"""
    1. **Tamanho da Amostra**: Com apenas {len(data)} observacoes (35 municipios × 2 anos), 
       o poder estatistico eh limitado. Recomenda-se ampliacao para 5+ anos.
    
    2. **Periodo Temporal Curto**: Apenas 2022-2023 podem nao capturar tendencias estruturais.
       Seria mais informativo analisar dados de 5-10 anos.
    
    3. **Restricao Geografica**: Analise limitada a regiao RIDE (Brasilia e adjacencias).
       Resultados podem nao ser generalizaveis para outras regioes.
    
    4. **Variaveis Confundidoras Nao Incluidas**:
       - Efetivo policial e investimento em seguranca publica
       - Educacao e oportunidades de emprego
       - Fatores demograficos (densidade populacional, idade media)
       - Tipo de urbanizacao (rural vs. urbano)
       - Dados de saude (SUS) poderiam servir como proxies
    
    5. **Causalidade vs. Correlacao**: Uma correlacao significativa nao implica 
       causalidade. A desigualdade pode estar correlacionada com criminalidade 
       atraves de mecanismos indiretos.
    
    6. **Agregacao de Crimes**: Total de vitimas pode mascarar padroes especificos 
       por tipo de crime. Analise separada por tipo poderia revelar diferentes padroes.
    
    7. **Qualidade dos Dados**: Possivel vies nos registros de crimes (subnotificacao, 
       praticas distintas de registro entre municipios).
    """)
    
    # Recomendações
    st.markdown("---")
    st.markdown("## Recomendacoes para Pesquisa Futura")
    
    st.markdown("""
    1. **Expandir Horizonte Temporal**: Incluir dados 2018-2024 para melhor compreensao de tendencias
    
    2. **Analise por Tipo de Crime**: Investigar se a relacao varia para crimes especificos 
       (violento vs. patrimonial, feminicidio, etc.)
    
    3. **Incluir Variaveis de Controle**: 
       - Indicadores de politicas publicas
       - Dados de educacao e emprego
       - Indicadores de saude (SUS)
    
    4. **Analise Espacial**: Considerar autocorrelacao espacial entre municipios vizinhos
    
    5. **Modelos Mais Sofisticados**: 
       - Regressao multipla com confundidores
       - Modelos nao-lineares (splines, GAM)
       - Abordagem Bayesiana para pequenas amostras
    
    6. **Validacao em Outras Regioes**: Replicar analise em outras regioes do Brasil 
       para avaliar generalizacao
    """)
    
    # Resumo técnico
    st.markdown("---")
    st.markdown("## Resumo Tecnico")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("N (observacoes)", len(data))
    with col2:
        st.metric("Correlacao Pearson", f"{correlation:.3f}")
    with col3:
        st.metric("P-valor", f"{p_value:.6f}")
    
    st.markdown(f"""
    - **Modelo**: Regressao Linear Simples (OLS)
    - **R² do Modelo**: {model.rsquared:.4f}
    - **R² Ajustado**: {model.rsquared_adj:.4f}
    - **F-estatistica**: {model.fvalue:.4f} (p = {model.f_pvalue:.6f})
    - **Validacao Cruzada**: K-Fold (k=5) recomendado para amostras pequenas
    """)




