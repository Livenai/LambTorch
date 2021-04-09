import contest
import telegram_debugger
from telegram_debugger import sendMSG
import traceback


# PARAMETROS

NUM_NETWORKS = 10


try:

    # Mensaje inicial
    contest.sendStarMSG()

    # Generamos las tareas
    contest.generateTasks(NUM_NETWORKS)

    # Iniciamos el entrenamiento de las tareas
    contest.trainRemainingTasks()

    # Mostramos el ranking
    contest.showRanking()

    # Mensaje final
    contest.sendFinalMSG()










except Exception as e:
    # Enviamos el error al admin
    sendMSG("ERROR", is_error=True)
    sendMSG(traceback.format_exc(), is_error=True)
