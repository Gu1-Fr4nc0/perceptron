# =======================================================================
# ARQUIVO: test_iris_v2.R
# DESCRIÇÃO: Script de teste para o Perceptron no dataset Iris, usando
#            as funções aprimoradas de visualização com ggplot2.
# =======================================================================

# Carrega as funções do nosso arquivo "biblioteca"
source("perceptron_v2.R")

cat("--- INICIANDO TESTE COM DATASET IRIS (VERSÃO GGPLOT2) ---\n")

# 1. PREPARAÇÃO DOS DADOS
# Seleciona apenas duas classes para binarização (problema mais simples e separável)
iris_subset <- subset(iris, Species %in% c("setosa", "versicolor"))

# Seleciona as features e a classe
X_iris <- iris_subset[, c("Petal.Length", "Petal.Width")]
y_iris <- ifelse(iris_subset$Species == "setosa", -1, 1)

# 2. TREINAMENTO
model <- train_perceptron(as.matrix(X_iris), y_iris, learning_rate = 0.1, seed = 42)

# 3. VISUALIZAÇÃO
cat("Gerando gráficos com ggplot2...\n")

# Gráfico de Erro
plot_error_history_gg(
  errors = model$errors, 
  epochs_ran = model$epochs_ran, 
  title = "Curva de Convergência - Iris (Setosa vs Versicolor)"
)

# Gráfico do Hiperplano Separador
plot_decision_boundary_gg(
  X = X_iris, 
  y = y_iris, 
  weights = model$weights, 
  title = "Hiperplano de Decisão - Iris (Setosa vs Versicolor)"
)

cat("--- ANÁLISE CONCLUÍDA ---\n")
