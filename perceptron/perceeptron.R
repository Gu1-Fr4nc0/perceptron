# perceptron.r
# Implementação do Perceptron Simples para Iniciação Científica

# Função de ativação degrau
step_function <- function(x) {
  return(ifelse(x >= 0, 1, -1))
}

# Função de treinamento do perceptron
perceptron.train <- function(train.set, weights = NULL, lrn.rate = 0.3, n.iter = 1000) {
  # iniciando variaveis de controle
  epochs <- 0
  error <- TRUE
  
  # se não passamos valores iniciais para os pesos sinápticos do perceptron,
  # eles serão gerados aleatoriamente com valores entre {-1, +1}
  if(is.null(weights)) {
    weights <- runif(ncol(train.set), -1, 1)
  }
  
  # mostrando os pesos gerados/recebidos
  cat("Pesos iniciais: ", round(weights, 4), "\n")
  
  # criamos um vetor para armazenar o erro médio por época
  avgErrorVec <- c()
  
  # variável de controla para especificar qual coluna do dataset contem a classe (target/label)
  class.id <- ncol(train.set)
  
  # enquanto houver erro em pelo menos um dos exemplos && número de épocas for
  # menor do que o número máximo de épocas que definimos
  while(error & epochs < n.iter) {
    error <- FALSE
    epochs <- epochs + 1
    avgError <- 0
    
    # vamos iterar sobre todos os exemplos do dataset (== época)
    for(i in 1:nrow(train.set)) {
      # acessamos o exemplo atual
      example <- as.numeric(train.set[i,])
      
      # calculamos a ativação do perceptron (spike)
      x <- example[-class.id]
      v <- as.numeric(x %*% weights[-1]) + weights[1]  # adicionando bias
      
      # gerando a saída do neurônio
      y <- ifelse(v >= 0, +1, -1)
      
      # vamos adicionar o erro quadrático do exemplo ao erro da época
      avgError <- avgError + ((example[class.id] - y)^2)
      
      # se a predição foi errada, temos que atualizar os pesos do perceptron
      if(example[class.id] != y) {
        # ajuste de pesos
        weights[-1] <- weights[-1] + lrn.rate * (example[class.id] - y) * x
        weights[1] <- weights[1] + lrn.rate * (example[class.id] - y)  # bias
        error <- TRUE
      }
    }
    
    # fim do ciclo da época, vamos armazenar o erro quadrático da época
    avgError <- avgError/nrow(train.set)
    avgErrorVec <- c(avgErrorVec, avgError)
    cat("Época: ", epochs, " - Avg Error = ", round(avgError, 4), "\n")
  }
  
  # vamos retornar um objeto com algumas informações para análises
  obj <- list(weights = weights, avgErrorVec = avgErrorVec, epochs = epochs)
  cat("\n* Finalizado depois de: ", epochs, " épocas\n")
  return(obj)
}

# Função de predição do perceptron
perceptron.predict <- function(test.set, weights) {
  # Adicionar bias se necessário
  if(ncol(test.set) == (length(weights) - 1)) {
    test.set <- cbind(1, test.set)
  }
  
  v <- as.numeric(test.set %*% weights)
  y <- ifelse(v >= 0, +1, -1)
  return(y)
}

# Função para calcular acurácia
calculate_accuracy <- function(y_true, y_pred) {
  return(mean(y_true == y_pred))
}

# Função para plotar curva de erro
plot_error_curve <- function(error_history, title) {
  df <- data.frame(epoch = 1:length(error_history), error = error_history)
  plot(df$epoch, df$error, type = "l", col = "blue", lwd = 2,
       main = title, xlab = "Época", ylab = "Erro Quadrático Médio")
  grid()
  points(df$epoch, df$error, col = "red", pch = 19, cex = 0.8)
}

# Função para plotar hiperplano
plot_decision_boundary <- function(X, y, weights, title) {
  plot(X[,1], X[,2], col = ifelse(y == 1, "red", "blue"), 
       pch = 19, cex = 2, main = title, xlab = "X1", ylab = "X2")
  
  # Calcular pontos para a linha de decisão
  if(length(weights) == 3) {
    x1_range <- range(X[,1])
    x2_values <- (-weights[1] - weights[2] * x1_range) / weights[3]
    lines(x1_range, x2_values, col = "green", lwd = 3)
  }
  
  legend("topright", legend = c("Classe +1", "Classe -1", "Hiperplano"), 
         col = c("red", "blue", "green"), pch = c(19, 19, NA), lty = c(NA, NA, 1))
}
