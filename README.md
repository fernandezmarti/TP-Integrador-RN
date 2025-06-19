Para la realización del trabajo se crearon los siguientes archivos:

4_comparacion BatchNorm.ipynb: Se buscan los 10 mejores modelos hasta el momento y se copian sus hiperparámetros. Luego se entrenan 10 modelos con esos hiperparámetros agregando Batchnorm. Finalmente se repite el proceso pero eliminando dropout. De este modo tenemos 10 modelos de control, 10 con BN y dropout, 10 BN sin dropout.

5_inicializacion pesos.ipynb: Se buscan los 10 mejores modelos hasta el momento y se copian sus hiperparámetros. En base a esos hiperparametros se crean 30 modelos, 10 para cada tipo de inicializacion (He, Xavier y uniform).

6_Analisis resultados.ipynb: Se pegan screenshots tanto del Mlfow UI como del tensorboard a modo de ANEXO. También se levantan algunos parametros y se extraen conclusiones pertinentes.

TP_integrador.ipynb: Acá se responden las preguntas teoricas.

Ademas se modifico el 0_EDA.ipynb y helper.py