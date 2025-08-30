# testAND.r
# Testa o Perceptron Simples com o dataset da porta lógica AND.

# Carrega as funções do perceptron
source("perceptron.r")

# --- 1. Preparação dos Dados ---
# O Perceptron deve funcionar, pois o problema é linearmente separável.
X1 <- c(0, 0, 1, 1)
X2 <- c(0, 1, 0, 1)
D_and <- c(-1, -1, -1, 1) # Saídas da porta AND (-1 para 0, +1 para 1)

# Cria o dataframe e adiciona a coluna de bias
dataset_and <- data.frame(Bias = 1, X1 = X1, X2 = X2, D = D_and)
print("Dataset AND:")
print(dataset_and)

# --- 2. Treinamento ---
obj_and <- perceptron.train(train.set = dataset_and, lrn.rate = 0.1)

# --- 3. Visualização dos Resultados ---

# Gráfico de Convergência (Época vs. Erro)
df_error_and <- data.frame(epoch = 1:obj_and$epochs, avgError = obj_and$avgErrorVec)
g_error_and <- ggplot(df_error_and, aes(x = epoch, y = avgError)) +
  geom_line(color = "blue") + geom_point(color = "blue") +
  labs(title = "Convergência do Perceptron no Dataset AND", x = "Época", y = "Erro Quadrático Médio") +
  theme_minimal()
print(g_error_and)


# Gráfico do Hiperplano
w0 <- obj_and$weights[1] # Peso do bias
w1 <- obj_and$weights[2] # Peso para X1
w2 <- obj_and$weights[3] # Peso para X2

# Equação da reta: w1*x1 + w2*x2 + w0 = 0  =>  x2 = -(w1/w2)*x1 - (w0/w2)
slope_and <- -w1 / w2
intercept_and <- -w0 / w2

dataPlot_and <- dataset_and
dataPlot_and$D <- as.factor(dataPlot_and$D)

g_hyperplane_and <- ggplot(dataPlot_and, aes(x = X1, y = X2, colour = D, shape = D)) +
  geom_point(size = 5) +
  theme_bw() +
  labs(title = "Hiperplano Separador - Dataset AND") +
  geom_abline(intercept = intercept_and, slope = slope_and, color = "red", linetype = "dashed")
print(g_hyperplane_and)
