# testIRIS.r
# Testa o Perceptron Simples com o dataset Iris, binarizando as classes.

# Carrega as funções do perceptron
source("perceptron.r")

# --- 1. Preparação dos Dados ---
# O problema será: Setosa vs. (Virginica + Versicolor)
# Usando Sepal.Width e Petal.Width, o problema é linearmente separável.

# Carrega o dataset
dataset <- iris
head(dataset)

# Análise Exploratória dos Dados
pairs(iris[,1:4], col = iris$Species, pch = 19, main = "Análise de Separação das Classes do Iris")

# Seleciona apenas duas features + a Classe
X <- dataset[, c(2, 4, 5)]

# Binariza o problema: setosa = +1, outros = -1
X$Species <- ifelse(X$Species == "setosa", +1, -1)

# Adiciona a coluna de bias
X <- cbind(Bias = 1, X)
colnames(X)[1] <- "bias"

print("Tabela de Frequência das Classes:")
print(table(X$Species))
print("Cabeçalho do Dataset Preparado:")
head(X)


# --- 2. Treinamento ---
obj_iris <- perceptron.train(train.set = X, lrn.rate = 0.1)

# --- 3. Visualização dos Resultados ---

# Gráfico de Convergência (Época vs. Erro)
df_error_iris <- data.frame(epoch = 1:obj_iris$epochs, avgError = obj_iris$avgErrorVec)
g_error_iris <- ggplot(df_error_iris, aes(x = epoch, y = avgError)) +
  geom_line(color = "darkgreen") + geom_point(color = "darkgreen") +
  labs(title = "Convergência do Perceptron no Dataset Iris", x = "Época", y = "Erro Quadrático Médio") +
  theme_minimal()
print(g_error_iris)

# Gráfico do Hiperplano
w0 <- obj_iris$weights[1]
w1 <- obj_iris$weights[2]
w2 <- obj_iris$weights[3]

slope_iris <- -w1 / w2
intercept_iris <- -w0 / w2

dataPlot_iris <- X
dataPlot_iris$Species <- as.factor(dataPlot_iris$Species)

g_hyperplane_iris <- ggplot(dataPlot_iris, aes(x = Sepal.Width, y = Petal.Width, colour = Species, shape = Species)) +
  geom_point(size = 3) +
  theme_bw() +
  labs(title = "Hiperplano Separador - Dataset Iris") +
  geom_abline(intercept = intercept_iris, slope = slope_iris, color = "red", linetype = "dashed")
print(g_hyperplane_iris)


# --- 4. Teste com novos exemplos (como no seu notebook) ---
test1 <- c(1, 0.4, 0.5) # bias, Sepal.Width, Petal.Width
res1 <- perceptron.predict(test.set = test1, weights = obj_iris$weights)
cat(" - Exemplo: ", test1, " - Classe predita: ", res1, "\n") # Esperado: -1

test2 <- c(1, 2.5, 0.3) # bias, Sepal.Width, Petal.Width
res2 <- perceptron.predict(test.set = test2, weights = obj_iris$weights)
cat(" - Exemplo: ", test2, " - Classe predita: ", res2, "\n") # Esperado: +1

# Adiciona os pontos de teste ao gráfico
g_final_iris <- g_hyperplane_iris +
  annotate("point", x = test1[2], y = test1[3], colour = "blue", size = 5, shape = 8) +
  annotate("point", x = test2[2], y = test2[3], colour = "purple", size = 5, shape = 8)
print(g_final_iris)
