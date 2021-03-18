"""
En este script se guardaran las utilidades necesarias para el control de
informacion de las redes:

    - Cargar y guardar redes en formato json
    - Gestion de la lista de redes ya creadas
    - Transformar una red en un json con su info
    - Crear una red a partir de un json con la info



net_info{
    hyperparams: {},
    net_layer_struct: {},


}

TODO: pensar que cosas se necesitan guardar de una red para poder luego crear otra red
a partir de esos datos del json. Crear todas las funciones para traducir de CL_trainer a json y viceversa.
Crear la funcion para cargar el gran json con todas las redes, asi como para guardarlo.

Despues de eso, modificar contest.py para que genere redes aleatorias, las entrene y las acumule en
el gran json, transformandolas a ese formato. Tienen que ser dos funciones diferentes, es decir, tiene que
haber una que sirva para añadir task al gran json (redes que aun no han sido entrenadas y por tanto no tienen metricas),
y debe haber otra funcion que recorra el json en busca de las task (redes no entrenadas) y las entrene, actualizando sus
datos y guardando las metricas del entrenamiento.

Hay que capturar el Ctrl-C para que si sucede, pare el entrenamiento que este haciendo y guarde los datos que tenia
resueltos hasta ese momento. aun que esto suponga perder los datos de la red que se estaba entrenando, al menos se guardan
los datos de las que ya se habian hecho.

Despues de esto deberiamos poder cargar el gran json y leerlo para entrenar las redes, asi como generar nuevas y acumularlas
para entrenar despues.

En contest deberia de haber dos funciones, una para entrenar las task, y otra para aladir nuevas task, pero es probable que
se necesite otra funcion para poder entrenar y generar nuevas redes automaticamente cuando haga falta.

por ultimo, habria que modificar el visor del ranking para que muestre todas las redes ordenadas con muuuuuuy pocos
datos, que muestre solo las ultimas N redes con algo mas de detalle o que muestre las ultimas N con todos los detalles.

Acordarse de arreglar lo de la memoria de la gpu, es decir, que se pueda limpiar la memoria despues de cada
entrenamiento, para poder entrenar de seguido, si no, se acaba la memori y lanza una excepcion.

"""
import json
