import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



def show_plot_and_save(history, regression=False, max_y_value=-1, just_save=False, save_name='fig'):
    """
    Muestra por pantalla una grafica con el historial del entrenamiento.

    Adaptado a la fase 1 y 2.
    """

    if regression:
        ent_loss = history['loss']
        val_loss = history['val_loss']
        ent_acc =  history['accuracy']
        val_acc =  history['val_accuracy']

        Gepochs = range(1, len(ent_loss) + 1)

        plt.style.use('dark_background')
        fig, axs = plt.subplots(2)
        fig.suptitle('Loss & Mean Err')

        if max_y_value >= 0:
            axs[0].set_ylim(top=max_y_value) # MAX_Y_LOSS


        axs[0].plot(Gepochs, ent_loss, 'lightcoral', label='Training Loss')
        axs[0].plot(Gepochs, val_loss, 'sandybrown', label='Test Loss')
        axs[1].plot(Gepochs, ent_acc, 'limegreen', label='Training Mean Error (Kg)')
        axs[1].plot(Gepochs, val_acc, 'greenyellow', label='Test Mean Error (Kg)')


        plt.xlabel('Epochs')
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0].legend()
        axs[1].legend()

        if just_save:
            plt.savefig(save_name + '.svg')
        else:
            plt.show()
    else:
        ent_loss = history['loss']
        val_loss = history['val_loss']
        ent_acc =  history['accuracy']
        val_acc =  history['val_accuracy']

        Gepochs = range(1, len(ent_loss) + 1)

        plt.style.use('dark_background')
        fig, axs = plt.subplots(2)
        fig.suptitle('Loss & Mean Err')

        if max_y_value >= 0:
            axs[0].set_ylim(top=max_y_value) # MAX_Y_LOSS


        axs[0].plot(Gepochs, ent_loss, 'lightcoral', label='Training Loss')
        axs[0].plot(Gepochs, val_loss, 'sandybrown', label='Test Loss')
        axs[1].plot(Gepochs, ent_acc, 'limegreen', label='Training Accuracy')
        axs[1].plot(Gepochs, val_acc, 'greenyellow', label='Test Accuracy')


        plt.xlabel('Epochs')
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0].legend()
        axs[1].legend()

        if just_save:
            plt.savefig(save_name + '.svg')
        else:
            plt.show()
