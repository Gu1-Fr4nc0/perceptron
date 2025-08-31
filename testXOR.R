# ==================================================================
# ARQUIVO: testXOR.r
# DESCRIÇÃO: Testa o Perceptron com o dataset da porta XOR.
#            Este problema NÃO é linearmente separável e não deve convergir.
# ==================================================================

library(ggplot2)
source("perceptron.r")

# Mude a semente para ver diferentes tentativas de separação
set.seed(42)

# Cria o dataset XOR
xor.data <- data.frame(
  Bias = 1,
  X1   = c(-1, -1, 1, 1),
  X2   = c(-1, 1, -1, 1),
  D    = c(-1, 1, 1, -1)
)

# Treina o modelo
cat("--- Treinando Perceptron para o problema XOR ---\n")
model <- perceptron.train(xor.data, lrn.rate = 0.1, n.iter = 100)

# --- GERAÇÃO DOS GRÁFICOS ---

# 1. Gráfico de Erro vs. Época
error_data <- data.frame(Epoca = 1:model$epochs, Erro = model$avgErrorVec)
error_plot <- ggplot(error_data, aes(x = Epoca, y = Erro)) +
  geom_line(color = "darkred") +
  geom_point(color = "darkred") +
  labs(title = "Performance do Treinamento (XOR)",
       subtitle = "Erro nunca chega a zero, pois o problema não é linearmente separável",
       x = "Época",
       y = "Erro Médio") +
  theme_minimal()

# 2. Gráfico do Hiperplano (mostrando a falha na separação)
w <- model$weights
slope <- -(w[2] / w[3])
intercept <- -(w[1] / w[3])

hyperplane_plot <- ggplot(xor.data, aes(x = X1, y = X2, color = as.factor(D))) +
  geom_point(size = 5) +
  labs(title = "Tentativa de Hiperplano para o Dataset XOR",
       subtitle = "Uma única reta não consegue separar as classes",
       x = "Entrada X1", y = "Entrada X2", color = "Classe") +
  geom_abline(intercept = intercept, slope = slope, linetype = "dashed", color = "black") +
  scale_color_manual(values = c("-1" = "red", "1" = "blue")) +
  theme_bw() +
  ylim(-0.5, 1.5) + xlim(-0.5, 1.5)

# Salva os gráficos em PDF
pdf("xor_plots.pdf")
print(error_plot)
print(hyperplane_plot)
dev.off()
