import tkinter as tk
import tools.exctract_data as ts
from clotcor.base import Prediccion

class Window:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Prevencion de delitos en Costa Rica")
        self.window.geometry("650x630")
        self.window.configure(bg="#0471A6") 
        self.frame = tk.Frame(self.window, padx=20, pady=20, bg='#061826')
        self.frame.grid()
        self.vars = {}
        self.labels = []
        self.option_menu()

    def option_menu(self):
        menu_configs = {
        'Hora': list(range(1, 25)),
        'DiaSemana': list(range(7)),
        'Mes': list(range(1, 13))
        }

        for space, (category, options) in enumerate(ts.unique_dict.items()):
            self.create_option_menu(category, options, space)

        for space, (category, options) in enumerate(menu_configs.items(), start=len(ts.unique_dict)):
            self.create_option_menu(category, options, space)

    def create_option_menu(self, category, options, space):
        self.vars[category] = tk.StringVar(self.window)
        self.vars[category].set(category) 

        option_menu = tk.OptionMenu(self.frame, self.vars[category], *options)
        option_menu.grid(column=1, row=space + 1, padx=10, pady=5, sticky="w")
        option_menu.config(bg='#3685B5', font=("Arial", 10), relief="groove")
   
    def window_content(self):
        self.labels = ["Delito", "Hora", "SubDelito", "Victima", "SubVictima", "Edad", "Sexo", 
                  "Nacionalidad", "Provincia", "Canton", "Distrito", "DiaSemana", "Mes"]
        
        for idx, text in enumerate(self.labels, start=1):
            label = tk.Label(self.frame, text=text, bg="#89AAE6", font=("Arial", 10, "bold"))
            label.grid(column=2, row=idx, padx=10, pady=5, sticky="w")

        quit_button = tk.Button(self.frame, text="Quit", command=self.window.destroy, 
                                bg="#AC80A0", fg="white", font=("Arial", 10, "bold"), relief="raised")
        quit_button.grid(column=1, row=0, pady=10, padx=10, sticky="w")
        predict_button = tk.Button(self.frame, text="Predecir", command=self.get_prediction, 
                                   bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), relief="raised")
        predict_button.grid(column=2, row=0, pady=10, padx=10, sticky="w")

    def get_prediction(self):
        try:
            self.prediccion = Prediccion()
            self.prediccion.run()
            new_data = {label: self.vars.get(label, tk.StringVar(self.window)).get() for label in self.labels}

            for key, value in new_data.items():
                if value is None or value == key:
                    raise ValueError(f"El campo '{key}' no ha sido seleccionado correctamente.")

            print(new_data)
            self.prediccion.predict_new_data(new_data)
        except ValueError as error:
            tk.messagebox.showerror("Error de entrada", str(error))

    def run(self):
        self.window_content()
        self.window.mainloop()

