# ==================================================================
# ARQUIVO: perceptron.r
# DESCRIÇÃO: Funções base para treinamento e predição de um 
#            modelo Perceptron Simples.
# ==================================================================

# -------------------------------------------------------------------------------------------------
# Perceptron.train 
#    - train.set: conjunto de treinamento (deve incluir a coluna de Bias)
#    - lrn.rate:  taxa de aprendizado
#    - n.iter:    número máximo de iterações (épocas)
# -------------------------------------------------------------------------------------------------
perceptron.train <- function(train.set, lrn.rate = 0.3, n.iter = 100) {
  
  # Gera pesos sinápticos aleatórios entre -1 e +1
  num_features <- ncol(train.set) - 1
  weights <- runif(num_features, -1, 1)
  
  cat("Pesos iniciais: ", weights, "\n")
  
  # Variáveis de controle
  epochs <- 0
  error <- TRUE
  avgErrorVec <- c()
  class.id <- ncol(train.set)
  
  # Loop de treinamento
  while (error & epochs < n.iter) {
    error <- FALSE
    epochs <- epochs + 1
    totalError <- 0
    
    # --- ALTERAÇÃO PRINCIPAL AQUI ---
    # Embaralha a ordem dos dados para cada nova época
    shuffled_indices <- sample(nrow(train.set))
    
    # Itera sobre cada exemplo do dataset NA ORDEM EMBARALHADA
    for (i in shuffled_indices) {
      example <- as.numeric(train.set[i, ])
      features <- example[-class.id]
      expected_class <- example[class.id]
      
      # Calcula a ativação
      activation <- sum(features * weights)
      
      # Gera a saída do neurônio
      predicted_class <- ifelse(activation >= 0, +1, -1)
      
      # Calcula o erro
      current_error <- expected_class - predicted_class
      totalError <- totalError + (current_error^2)
      
      # Se a predição estiver errada, atualiza os pesos
      if (current_error != 0) {
        weights <- weights + lrn.rate * current_error * features
        error <- TRUE
      }
    }
    
    # Armazena o erro quadrático médio da época
    avgError <- totalError / nrow(train.set)
    avgErrorVec <- c(avgErrorVec, avgError)
    cat("Época: ", epochs, " - Erro Médio = ", avgError, "\n")
  }
  
  cat("\n* Treinamento finalizado após ", epochs, " épocas.\n")
  
  # Retorna o modelo treinado e o histórico de erro
  obj <- list(weights = weights, avgErrorVec = avgErrorVec, epochs = epochs)
  return(obj)
}

# -----------------------------------------------------------------
# Perceptron.predict: faz a predição para um novo exemplo
# -----------------------------------------------------------------
perceptron.predict <- function(test.set, weights) {
  activation <- sum(test.set * weights)
  predicted_class <- ifelse(activation >= 0, +1, -1)
  return(predicted_class)
}
