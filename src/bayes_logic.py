import json
import os
class TrollBrain:
    def __init__(self, data_path="data/memoria_ia.json"):
        self.data_path = data_path
        #Dos siccionarios: Clave=Palabra, Valor=Frecuencia
        self.vocab_toxico = {}
        self.vocab_pro = {}
        self.cargar_memoria()

    def aprender(self, frase, categoria):
        """"Descompone la frase y guarda las palabras en la categoria correcta"""
        palabras = frase.lower().split()

        for palabra in palabras:
            #Filtro basico ignorar palabras de menos 2 letras
            if len(palabra) >2:
                if categoria == "toxico":
                    #Si existe suma 2, si no existe empieza en 0 y suma 1
                    self.vocab_toxico[palabra] = self.vocab_toxico.get(palabra,0)+1
                else:
                    self.vocab_pro[palabra] = self.vocab_pro.get(palabra,0)+1

        self.guardar_memoria()
    
    def predecir(self, frase):
        """Calcula los puntos para cada bando(Algoritmo Naive Bayes Simplificado)"""
        score_toxico = 0
        score_pro = 0
        detalles = [] #Para explicar por qué tomó la decisión
        palabras = frase.lower().split()

        for palabra in palabras:
            pt = self.vocab_toxico.get(palabra,0)
            pp = self.vocab_pro.get(palabra, 0)
            score_toxico += pt
            score_pro += pp

            if pt > 0 or pp >0:
                detalles.append(f"'{palabra}' (T:{pt} /  A:{pp})")

        return score_toxico, score_pro, detalles
    
    def guardar_memoria(self):
        datos = {
            "toxico": self.vocab_toxico,
            "pro": self.vocab_pro
        }
        #Asegurar que el directorio existe
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        with open(self.data_path, 'w') as f:
            json.dump(datos, f)

    def cargar_memoria(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                datos = json.load(f)
                self.vocab_toxico = datos.get("toxico", {})
                self.vocab_pro = datos.get("pro", {})