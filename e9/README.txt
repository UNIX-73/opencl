He intentado utilizar el mismo contexto de OpenGL siguiendo lo que entiendo en los apuntes,
pero falla. Por lo que he estado leyendo, (https://community.khronos.org/t/multi-platform-contexts/2916).
Por lo que pone, no está permitido crear un contexto que utiliza distintas plataformas, por eso, no puedo crear
en el mismo cntexto una cpu y gpu, al menos hasta donde llega mi entendimiento. He dejado el código en el que intentaba
crear el mismo contexto guardado como vadd_c_same_context_attempt.c. Diría que de la manera que estoy haciendolo
en el código que se compila (vadd_c.c) también se ejecuta en paralelo, por eso 