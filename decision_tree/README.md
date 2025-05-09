# DecisionTreeID3

Implementação pura do algoritmo ID3 para atributos **categóricos**.

## 📝 Algoritmo ID3 em detalhes

1. **Node** (`Node`):  
   - `feature` (índice)  
   - `category` (valor testado)  
   - `left` / `right` (subárvores)  
   - `value` (classe majoritária em nós-folha)

2. **Fit** (`fit(X, y)`):  
   - Converte `X`/`y` para `ndarray`  
   - Garante `X` em 2D  
   - Define `n_features` (padrão = todas)  
   - Chama `_grow_tree`

3. **Crescimento recursivo** (`_grow_tree`):  
   - Critérios de parada:  
     - `n_samples < min_samples_split`  
     - `depth >= max_depth`  
     - todas as instâncias têm mesma classe  
     - **melhor ganho ≤ 0**  
   - Em cada nó:  
     - Amostra aleatória de `n_features` índices  
     - Para cada `(feature, categoria)`:  
       - Particiona em `==` vs `!=`  
       - Calcula **Information Gain** (diferença de entropia)  
     - Escolhe `best_feat` + `best_cat`  
     - Se `best_gain > 0`, cria `Node` e recursa para `left`/`right`; senão, nó-folha.

4. **Entropia** (`_entropy(y)`):  
   - Usa `Counter` para contar frequências  
   - \(-\sum p_i \log p_i\)

5. **Information Gain** (`_information_gain(col, y, cat)`):  
   - Entropia do pai − entropia ponderada dos filhos

6. **Split categórico** (`_split(col, cat)`):  
   - Igual vs diferente

7. **Predict** (`predict(X)`):  
   - Travessa cada amostra do nó raiz até nó-folha seguindo `==`/`!=`

## 💻 Exemplo de Uso

```py
from id3_decision_tree import DecisionTreeID3

clf = DecisionTreeID3(max_depth=5, min_samples_split=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)