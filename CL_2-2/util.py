import numpy as np
from colored import fg




# ------------- PROGRESS BAR  -------------------------------------------------------------------
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='■', printEnd="\r", color=118, print_finish='\n'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        color       - Optional  : progress bar color
    """
    if type(color) == int:
        color = fg(color)
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = color + fill * filledLength + fg(15) + "-" * (length - filledLength)
    print("%s |%s|" % (prefix, bar) + color + "%s" % percent + fg(15) + "%% %s" % suffix, end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print(print_finish, end='')




def cropWithZeros(in_array, x, y, h, w):
    """ Funcion que rellena con 0 toda la imagen menos el area seleccionada. """
    in_array = np.array(in_array)
    shape = in_array.shape
    crop = in_array[y:y + h, x:x + w]
    bx = shape[0] - x - h
    by = shape[1] - y - w
    padding = ((x, bx), (y, by))
    return np.pad(crop, padding)
