import contest
import telegram_debugger
from telegram_debugger import sendMSG


# PARAMETROS

NUM_NETWORKS = 10



# Generamos las tareas
contest.generateTasks(NUM_NETWORKS)

# Iniciamos el entrenamiento de las tareas
contest.trainRemainingTasks()

# Mostramos el ranking
contest.showRanking()

# Mensaje final
contest.sendFinalMSG()
