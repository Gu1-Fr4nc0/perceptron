# testXOR.r
# Testa o Perceptron Simples com o dataset da porta lógica XOR.

# Carrega as funções do perceptron
source("perceptron.r")

# --- 1. Preparação dos Dados ---
# O Perceptron NÃO deve funcionar, pois o problema não é linearmente separável.
X1 <- c(0, 0, 1, 1)
X2 <- c(0, 1, 0, 1)
D_xor <- c(-1, 1, 1, -1) # Saídas da porta XOR

# Cria o dataframe e adiciona a coluna de bias
dataset_xor <- data.frame(Bias = 1, X1 = X1, X2 = X2, D = D_xor)
print("Dataset XOR:")
print(dataset_xor)

# --- 2. Treinamento ---
# O treinamento atingirá o n.iter máximo, pois o erro nunca será zero.
obj_xor <- perceptron.train(train.set = dataset_xor, lrn.rate = 0.1, n.iter = 100)

# --- 3. Visualização dos Resultados ---

# Gráfico de Convergência (Época vs. Erro)
df_error_xor <- data.frame(epoch = 1:obj_xor$epochs, avgError = obj_xor$avgErrorVec)
g_error_xor <- ggplot(df_error_xor, aes(x = epoch, y = avgError)) +
  geom_line(color = "red") + geom_point(color = "red") +
  labs(title = "Convergência do Perceptron no Dataset XOR (Não Converge)", x = "Época", y = "Erro Quadrático Médio") +
  theme_minimal()
print(g_error_xor)


# Gráfico do Hiperplano
w0 <- obj_xor$weights[1]
w1 <- obj_xor$weights[2]
w2 <- obj_xor$weights[3]

slope_xor <- -w1 / w2
intercept_xor <- -w0 / w2

dataPlot_xor <- dataset_xor
dataPlot_xor$D <- as.factor(dataPlot_xor$D)

g_hyperplane_xor <- ggplot(dataPlot_xor, aes(x = X1, y = X2, colour = D, shape = D)) +
  geom_point(size = 5) +
  theme_bw() +
  labs(title = "Hiperplano - Dataset XOR (Não Separável)") +
  geom_abline(intercept = intercept_xor, slope = slope_xor, color = "red", linetype = "dashed")
print(g_hyperplane_xor)
