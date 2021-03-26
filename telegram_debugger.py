"""
Este escript manejara el bot de telegram para la depuracion y el control remoto
"""
from colored import fg
import telepot, os
from emoji import emojize


parent_folder = os.path.abspath(os.path.dirname(__file__))

########################  Ejecucion a hacer import  ########################

# Obtenemos el token
try:
    token_path = os.path.join(parent_folder, "telegram_token")
    token = open(token_path).readline().replace("\n", "")
    print("Token encontrado:  \""+ fg(2) + token + fg(15) + "\"")
except:
    print(fg(15) + "[!]" + fg(1) + " Token no encotrado." + fg(15))
    token = None



# Intenamos conectar el bot. Si no puede por lo que sea (por ejemplo, si no
# hay internet), avisa por pantalla y marca a True el booleano JUST_PRINT

JUST_PRINT = False

my_bot = None
try:
    my_bot = telepot.Bot(token)
except:
    print(fg(208) + "[!] Error al iniciar bot de telegram. Imprimiendo solo por pantalla [!]" + fg(15))
    JUST_PRINT = True




#################################  Funciones  #################################

def sendMSG(msg, is_warning = False, is_error = False, dont_print = False):
    """
    Procesa un mensaje acorde a la situacion.

    Si el bot de telegram esta inicializado, envia un mensaje de debug
    al admin e imprime el error por pantalla.

    Si no, solo imprime el error por pantalla.

    Hace auto emojize, por lo que acepta emojis.
    """
    # Formateamos en funcion del tipo de mensaje
    if is_error:
        t_msg = emojize(":no_entry: " + msg + " :no_entry:")
        p_msg = fg(9) + msg + fg(15)
    elif is_warning:
        t_msg = emojize(":warning: " + msg + " :warning:")
        p_msg = fg(226) + msg + fg(15)
    else:
        t_msg = emojize(msg)
        p_msg = fg(154) + msg + fg(15)


    if JUST_PRINT == False:
        # Obtenemos el id del admin
        admin_id_path = os.path.join(parent_folder, "telegram_admin_id")
        admin_id = open(admin_id_path, "r").read()

        my_bot.sendMessage(chat_id=admin_id, text=t_msg)

    # Imprimimos por pantalla
    if dont_print == False:
        print(emojize(p_msg))
