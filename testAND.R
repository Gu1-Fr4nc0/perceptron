# ==================================================================
# ARQUIVO: testAND.r
# DESCRIÇÃO: Testa o Perceptron com o dataset da porta AND.
#            Este problema é linearmente separável e deve convergir.
# ==================================================================

#install.packages("ggplot2")
library(ggplot2)
source("perceptron.r")

set.seed(42)

# --- ALTERAÇÃO PRINCIPAL AQUI ---
# Cria o dataset AND usando -1 para "falso" e +1 para "verdadeiro"
and.data <- data.frame(
  Bias = 1,
  X1   = c(-1, -1, 1, 1),
  X2   = c(-1, 1, -1, 1),
  D    = c(-1, -1, -1, 1)
)

# Treina o modelo Perceptron
cat("--- Treinando Perceptron para o problema AND ---\n")
model <- perceptron.train(and.data, lrn.rate = 0.01, n.iter = 100)

# --- GERAÇÃO DOS GRÁFICOS ---
error_data <- data.frame(Epoca = 1:model$epochs, Erro = model$avgErrorVec)
error_plot <- ggplot(error_data, aes(x = Epoca, y = Erro)) +
  geom_line(color = "steelblue") + geom_point(color = "steelblue") +
  labs(title = "Performance do Treinamento (AND)",
       subtitle = "Erro Quadrático Médio por Época",
       x = "Época", y = "Erro Médio") +
  theme_minimal()

w <- model$weights
slope <- -(w[2] / w[3])
intercept <- -(w[1] / w[3])

hyperplane_plot <- ggplot(and.data, aes(x = X1, y = X2, color = as.factor(D))) +
  geom_point(size = 5) +
  labs(title = "Hiperplano Gerado para o Dataset AND",
       x = "Entrada X1", y = "Entrada X2", color = "Classe") +
  geom_abline(intercept = intercept, slope = slope, linetype = "dashed", color = "black") +
  scale_color_manual(values = c("-1" = "red", "1" = "blue")) +
  theme_bw() +
  ylim(-2, 2) + xlim(-2, 2)

pdf("and_plots.pdf")
print(error_plot)
print(hyperplane_plot)
dev.off()
