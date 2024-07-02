<h1 align="center">
  Predição de AVC através de aprendizado de máquina
</h1>


## 💻 Projeto
O objetivo deste repositório é servir como um local centralizado para armazenar scripts relacionados ao tratamento de dados e à geração de gráficos e resultados. Ele está focado na análise sistemática de dados de pacientes com Acidente Vascular Cerebral (AVC), visando realizar previsões precisas através de técnicas de classificação. Os resultados da pasta weka-results foram gerados através do programa Weka, na versão 3.8.6.

## Integrantes do grupo

[Marco Antonio](https://github.com/scush989898) <br>
[Vitor Buss](https://github.com/VitorManoelBuss) <br>
[João Pedro](https://github.com/kinkbaldhead) <br>

Tipos de Algorítmos de classificação utilizados:

- Rules
- Trees

Algorítmos utilizados:

- Rules
    - DecisionTable
    - JRip
    - OneR
    - PART
    - ZeroR
  
- Trees
    - DecisionStump
    - HoeffdingTree
    - J48
    - LMT
    - Random Forest
    - Random Tree
    - REPTree

## ✨ Tecnologias

- [X] Python
- [X] Venv
- [X] Weka


# Instruções:

### Crie o ambiente virtual
```
python -m venv venv
```
### Ative o venv
```bash
# linux: 

source venv/bin/activate

# windows: 

.\venv\Scripts\activate

```

### Instale as dependências 
```
pip install -r requirements.txt
```

### Executando os scripts
```
python data_cleaner.py
```
```
python graph_generator.py
```

