# CC6409-Flask

Ejemplo de deploy ML en Flask para el curso CC6409

El código para la API de Flask (carpeta `pytorch-api`) forma parte de un tutorial propiedad de Pytorch [Link al tutorial](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)

El repositorio se compone de dos proyectos en Flask:

- `classifier-app`, que contiene una aplicación simple la cual recibe una
  imagen y devuelve su clase de ImageNet mediante el llamado a una API.
- `pytorch-api`, que corresponde a la API Rest. Dicho proyecto instancia un modelo DenseNet121
  preentrenado y lo utiliza para predecir la clase correspondiente a la imagen que recibe por POST,
  devolviendo un JSON con la clase predicha.

## Instrucciones

- Instalar librerías necesarias.
  
  ```bash
   conda install flask
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch 
  ```

- Nótese que el comando para instalar pytorch podría variar según su sistema operativo. 
  Se recomienda visitar [Pytorch - Install](https://pytorch.org) y copiar el comando que corresponda según su configuración.

- En un primer terminal, levantar la API.
  
  ```bash
   cd pytorch-api
   python main.py
  ```

- En un segundo terminal, levantar la app de frontend:
  
  ```bash
   cd classifier-app
   python main.py
  ```

- Por defecto, la API quedará en el puerto 5001 y el front en el puerto 5000.
