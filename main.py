import contest


# PARAMETROS

NUM_NETWORKS = 10



# Generamos las tareas
contest.generateTasks(NUM_NETWORKS)

# Iniciamos el entrenamiento de las tareas
contest.trainRemainingTasks()

# Mostramos el ranking
contest.printRankingAux()
