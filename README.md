# Random Forest From Scratch

Uma implementação didática de Random Forest em Python, sem dependências externas além de NumPy e Pandas.

## 📖 Sobre

- **O que é Random Forest?**  
  Conjunto de árvores de decisão treinadas em diferentes _bootstraps_ do mesmo conjunto de dados, com subamostragem de features. Combinação por votação majoritária que reduz variância e melhora robustez.
- **Por que “from scratch”?**  
  Para entender cada passo do algoritmo clássico de bagging e feature‐sampling, sem mistérios de bibliotecas.

## 🚀 Principais Funcionalidades

- Bootstrap sampling  
- Construção de múltiplas Decision Trees via ID3 (categórico)  
- Votação majoritária  
- Ajuste de hiperparâmetros: número de árvores, profundidade, número de features por split