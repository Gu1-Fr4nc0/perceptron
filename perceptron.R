# Adicionando um seed para reproducibilidade dos exemplos
set.seed(42)

# Carregando o pacote para gerar gráficos
library("ggplot2")

# -------------------------------------------------------------------------------------------------
# perceptron.train
#    - train.set: conjunto de treinamento com colunas de features e a última coluna como classe (+1 ou -1)
#    - weights:   pesos sinápticos iniciais (opcional)
#    - lrn.rate:  taxa de aprendizado para o ajuste dos pesos
#    - n.iter:    número máximo de iterações (épocas) para o treinamento
# -------------------------------------------------------------------------------------------------
perceptron.train <- function(train.set, weights = NULL, lrn.rate = 0.3, n.iter = 1000) {
  
  # Iniciando variáveis de controle
  epochs <- 0
  error  <- TRUE
  
  # Se os pesos iniciais não forem fornecidos, eles serão gerados aleatoriamente com valores entre [-1, +1]
  if(is.null(weights)) {
    weights <- runif(ncol(train.set) - 1, -1, 1)
  }
  
  # Mostrando os pesos iniciais
  cat("Pesos iniciais: ", weights, "\n")
  
  # Vetor para armazenar o erro médio por época
  avgErrorVec <- c()
  
  # Identifica a coluna da classe (target/label)
  class.id <- ncol(train.set)
  
  # Loop de treinamento: continua enquanto houver erro e o número de épocas for menor que o máximo
  while(error & epochs < n.iter) {
    error  <- FALSE
    epochs <- epochs + 1
    avgError <- 0
    
    # Itera sobre todos os exemplos do dataset (uma época)
    for(i in 1:nrow(train.set)) {
      # Exemplo atual
      example <- as.numeric(train.set[i,])
      
      # Calcula a ativação do perceptron (combinação linear)
      x <- example[-class.id]
      v <- as.numeric(x %*% weights)
      
      # Gera a saída do neurônio (função de ativação degrau)
      y <- ifelse(v >= 0, +1, -1)
      
      # Acumula o erro quadrático da época
      avgError <- avgError + ((example[class.id] - y)^2)
      
      # Se a predição foi errada, atualiza os pesos
      if(example[class.id] != y) {
        weights <- weights + lrn.rate * (example[class.id] - y) * example[-class.id]
        error <- TRUE
      }
    }
    
    # Ao final da época, calcula o erro médio e o armazena
    avgError <- avgError / nrow(train.set)
    avgErrorVec <- c(avgErrorVec, avgError)
    cat("Época: ", epochs, " - Erro Médio = ", avgError, "\n")
  }
  
  # Retorna um objeto com os resultados do treinamento
  obj <- list(weights = weights, avgErrorVec = avgErrorVec, epochs = epochs)
  
  cat("\n* Finalizado depois de : ", epochs, " épocas\n")
  return(obj)
}

# -----------------------------------------------------------------
# perceptron.predict
#    - test.set: um único exemplo ou um conjunto de exemplos de teste
#    - weights:  pesos sinápticos obtidos no treinamento
# -----------------------------------------------------------------
perceptron.predict <- function(test.set, weights) {
  v <- as.numeric(test.set %*% weights)
  y <- ifelse(v >= 0, +1, -1)
  return(y)
}
