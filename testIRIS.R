# ==================================================================
# ARQUIVO: testIRIS.r
# DESCRIÇÃO: Testa o Perceptron com o dataset Iris.
#            - Seleciona 2 colunas: Sepal.Length e Sepal.Width
#            - Binariza a classe: setosa (1) vs. não-setosa (-1)
# ==================================================================

library(ggplot2)
source("perceptron.r")

# Mude a semente para testar diferentes inicializações
set.seed(42)

# Carrega e prepara o dataset Iris
data(iris)

# 1. Seleciona as duas colunas de features
iris_subset <- iris[, c("Sepal.Length", "Sepal.Width", "Species")]
names(iris_subset) <- c("X1", "X2", "Species")

# 2. Binariza a classe
iris_subset$D <- ifelse(iris_subset$Species == "setosa", 1, -1)

# Cria o dataset final para o treinamento (sem a coluna 'Species')
iris.train.data <- data.frame(
  Bias = 1,
  X1 = iris_subset$X1,
  X2 = iris_subset$X2,
  D = iris_subset$D
)

# Treina o modelo
cat("--- Treinando Perceptron para o problema Iris (setosa vs. não-setosa) ---\n")
model <- perceptron.train(iris.train.data, lrn.rate = 0.005, n.iter = 100)

# --- GERAÇÃO DOS GRÁFICOS ---

# 1. Gráfico de Erro vs. Época
error_data <- data.frame(Epoca = 1:model$epochs, Erro = model$avgErrorVec)
max_epochs <- model$epochs
breaks <- if(max_epochs <= 20) 1:max_epochs else unique(round(seq(1, max_epochs, length.out = 10)))

error_plot <- ggplot(error_data, aes(x = Epoca, y = Erro)) +
  geom_line(color = "darkgreen") +
  geom_point(color = "darkgreen") +
  scale_x_continuous(breaks = breaks) + # Eixo X com inteiros
  labs(title = "Performance do Treinamento (Iris)",
       subtitle = "Classificação: Setosa vs. Não-Setosa",
       x = "Época",
       y = "Erro Médio") +
  theme_minimal()

# 2. Gráfico do Hiperplano
w <- model$weights
slope <- -(w[2] / w[3])
intercept <- -(w[1] / w[3])

hyperplane_plot <- ggplot(iris.train.data, aes(x = X1, y = X2, color = as.factor(D), shape = as.factor(D))) +
  geom_point(size = 3) +
  labs(title = "Hiperplano Gerado para o Dataset Iris",
       subtitle = "Separação: Setosa (1) vs. Não-Setosa (-1)",
       x = "Sepal.Length", y = "Sepal.Width", color = "Classe", shape = "Classe") +
  geom_abline(intercept = intercept, slope = slope, linetype = "dashed", color = "black") +
  scale_color_manual(values = c("-1" = "red", "1" = "blue"),
                     labels = c("Não-Setosa", "Setosa")) +
  scale_shape_manual(values = c("-1" = 16, "1" = 17), # 16 = círculo, 17 = triângulo
                     labels = c("Não-Setosa", "Setosa")) +
  theme_bw()

# Salva os gráficos em PDF
pdf("iris_plots.pdf")
print(error_plot)
print(hyperplane_plot)
dev.off()
