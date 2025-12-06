# Interface graphique du projet de prédiction de salaire

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import threading
import datetime
import os
import sys
import pandas as pd
import numpy as np

from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config_paths import (
    BASE_DIR,
    NETTOYAGE_SCRIPT,
    TRAIN_SCRIPT,
    BEST_MODEL_PATH,
    CLEANED_DATA_PATH,
    LOGO_PATH,
)
from model_utils import load_model, load_clean_dataset, FEATURES, TARGET


class PipelineApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Projet TAA - Prédiction de salaire des développeurs")
        self.geometry("1100x650")
        self.minsize(900, 550)

        self.configure(bg="#1e1e1e")

        self.model = None
        self.predict_window = None
        self.combo_values = None

        self.figure = None
        self.canvas = None
        self.graph_df = None

        self.var_graph_type = tk.StringVar(
            value="Distribution des salaires (global)"
        )

        self._setup_style()
        self._build_ui()
        self._build_menu()

        self.log("Interface prête. Clique sur un bouton pour lancer une étape.")

    # Style général de l’interface
    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        bg_main = "#1e1e1e"
        bg_panel = "#252526"
        accent = "#24576B"
        text_color = "#ffffff"
        btn_fg = "#000000"

        style.configure("Main.TFrame", background=bg_main)
        style.configure("Panel.TFrame", background=bg_panel)

        style.configure(
            "Title.TLabel",
            background=bg_main,
            foreground=text_color,
            font=("Segoe UI", 16, "bold"),
        )

        style.configure(
            "Header.TLabel",
            background=bg_main,
            foreground=text_color,
            font=("Segoe UI", 12, "bold"),
        )

        style.configure(
            "TButton",
            font=("Segoe UI", 11),
            padding=8,
        )

        style.configure(
            "Primary.TButton",
            background=accent,
            foreground=btn_fg,
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#1B4353")],
        )

        style.configure(
            "Subtitle.TLabel",
            background=bg_panel,
            foreground=text_color,
            font=("Segoe UI", 11, "bold"),
        )

    # Barre de menus
    def _build_menu(self):
        menubar = tk.Menu(self)

        menu_pipeline = tk.Menu(menubar, tearoff=0)
        menu_pipeline.add_command(
            label="Nettoyer le dataset", command=self.on_clean_clicked
        )
        menu_pipeline.add_command(
            label="Entraîner (baseline)", command=self.on_train_clicked
        )
        menu_pipeline.add_separator()
        menu_pipeline.add_command(
            label="Lancer tout le pipeline", command=self.on_run_all_clicked
        )
        menu_pipeline.add_separator()
        menu_pipeline.add_command(label="Quitter", command=self.destroy)

        menu_view = tk.Menu(menubar, tearoff=0)
        menu_view.add_command(
            label="Afficher l’onglet Logs", command=self._show_logs_tab
        )
        menu_view.add_command(
            label="Afficher l’onglet Graphiques", command=self._show_graphs_tab
        )
        menu_view.add_command(
            label="Actualiser les graphiques", command=self.update_graphs
        )

        menu_help = tk.Menu(menubar, tearoff=0)
        menu_help.add_command(label="À propos", command=self._show_about)

        menubar.add_cascade(label="Pipeline", menu=menu_pipeline)
        menubar.add_cascade(label="Affichage", menu=menu_view)
        menubar.add_cascade(label="Aide", menu=menu_help)

        self.config(menu=menubar)

    # Fenêtre À propos
    def _show_about(self):
        messagebox.showinfo(
            "À propos",
            "Projet TAA - Prédiction de salaire des développeurs\n"
            "Réalisé par ARAB ASSALAS & HALIT DOUNIA\n"
            "Étudiants en M1 IBD",
        )

    # Construction des éléments principaux
    def _build_ui(self):
        main_frame = ttk.Frame(self, style="Main.TFrame", padding=10)
        main_frame.pack(fill="both", expand=True)

        header_frame = ttk.Frame(main_frame, style="Main.TFrame")
        header_frame.pack(fill="x", pady=(0, 5))

        title = ttk.Label(
            header_frame,
            text="Projet TAA - Prédiction de salaire des développeurs",
            style="Title.TLabel",
            anchor="w",
        )
        title.grid(row=0, column=0, sticky="w")

        subtitle = ttk.Label(
            header_frame,
            text="Projet de ARAB ASSALAS & HALIT DOUNIA – Étudiants en M1 IBD",
            style="Header.TLabel",
            anchor="w",
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.logo_label = ttk.Label(header_frame, style="Main.TFrame")
        self.logo_label.grid(
            row=0, column=1, rowspan=2, sticky="e", padx=(20, 0)
        )

        header_frame.columnconfigure(0, weight=1)
        header_frame.columnconfigure(1, weight=0)

        self._load_logo()

        buttons_frame = ttk.Frame(main_frame, style="Main.TFrame")
        buttons_frame.pack(fill="x", pady=(5, 10))

        btn_clean = ttk.Button(
            buttons_frame,
            text="1 | Nettoyer le dataset",
            command=self.on_clean_clicked,
        )
        btn_train = ttk.Button(
            buttons_frame,
            text="2 | Entraîner les modèles",
            command=self.on_train_clicked,
        )
        btn_predict = ttk.Button(
            buttons_frame,
            text="3 | Prédire un salaire",
            command=self.on_predict_clicked,
        )
        btn_all = ttk.Button(
            buttons_frame,
            text="Lancer tout le pipeline",
            style="Primary.TButton",
            command=self.on_run_all_clicked,
        )
        btn_quit = ttk.Button(
            buttons_frame,
            text="Quitter",
            command=self.destroy,
        )

        for col in range(5):
            buttons_frame.columnconfigure(col, weight=1)

        btn_clean.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        btn_train.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        btn_predict.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        btn_all.grid(row=0, column=4, padx=5, pady=5, sticky="ew")
        btn_quit.grid(row=1, column=4, padx=(5, 0), pady=5, sticky="e")

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True, pady=(5, 0))

        self.logs_tab = ttk.Frame(self.notebook, style="Main.TFrame")
        self.notebook.add(self.logs_tab, text="Console / Logs")

        self.graphs_tab = ttk.Frame(self.notebook, style="Main.TFrame")
        self.notebook.add(self.graphs_tab, text="Graphiques")

        logs_frame = ttk.Frame(self.logs_tab, style="Panel.TFrame", padding=10)
        logs_frame.pack(fill="both", expand=True, pady=(5, 0))

        lbl_logs = ttk.Label(
            logs_frame, text="Logs :", style="Subtitle.TLabel"
        )
        lbl_logs.pack(anchor="w")

        text_frame = ttk.Frame(logs_frame, style="Panel.TFrame")
        text_frame.pack(fill="both", expand=True, pady=(5, 0))

        self.text = tk.Text(
            text_frame,
            wrap="word",
            bg="#1e1e1e",
            fg="#f0f0f0",
            insertbackground="#1B4353",
            font=("Consolas", 10),
            relief="flat",
        )
        scrollbar = ttk.Scrollbar(text_frame, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)

        self.text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        graph_top = ttk.Frame(self.graphs_tab, style="Main.TFrame")
        graph_top.pack(fill="x", pady=(8, 4), padx=5)

        lbl_graph = ttk.Label(
            graph_top,
            text="Visualisation des données et du modèle",
            style="Header.TLabel",
        )
        lbl_graph.pack(side="left", padx=(5, 10))

        graph_options = [
            "Distribution des salaires (global)",
            "Salaire moyen par pays (Top 10)",
            "Salaire moyen par DevType (Top 10)",
            "Salaire moyen vs YearsCoding",
            "Salaire moyen vs YearsCodingProf",
            "Salaire moyen par niveau d’étude",
            "Performance modèle : Réel vs prédictions (test)",
            "Performance modèle : Distribution des résidus (test)",
            "Performance modèle : Résidus vs prédictions (test)",
            "Performance modèle : Importance des variables (Top 20)",
        ]
        self.graph_combo = ttk.Combobox(
            graph_top,
            textvariable=self.var_graph_type,
            values=graph_options,
            state="readonly",
            width=55,
        )
        self.graph_combo.pack(side="left", padx=(0, 10))
        self.graph_combo.current(0)

        btn_refresh_graph = ttk.Button(
            graph_top,
            text="Actualiser les graphiques",
            command=self.update_graphs,
        )
        btn_refresh_graph.pack(side="right", padx=(0, 5))

        graph_frame = ttk.Frame(self.graphs_tab, style="Panel.TFrame")
        graph_frame.pack(fill="both", expand=True, pady=(0, 5), padx=5)

        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Prêt")
        status_bar = ttk.Label(
            self,
            textvariable=self.status_var,
            anchor="w",
            padding=5,
        )
        status_bar.pack(fill="x", side="bottom")

    # Chargement du logo de l’université
    def _load_logo(self):
        if not LOGO_PATH.exists():
            return
        try:
            img = Image.open(LOGO_PATH)
            img = img.resize((80, 80), Image.LANCZOS)
            self.logo_img = ImageTk.PhotoImage(img)
            self.logo_label.configure(image=self.logo_img)
        except Exception as e:
            self.log(f"[WARN] Impossible de charger le logo : {e}")

    # Gestion des onglets
    def _show_logs_tab(self):
        self.notebook.select(self.logs_tab)

    def _show_graphs_tab(self):
        self.notebook.select(self.graphs_tab)

    # Gestion de la zone de logs
    def log(self, message: str):
        timestamp = datetime.datetime.now().strftime("[%H:%M:%S] ")
        self.text.insert("end", timestamp + message + "\n")
        self.text.see("end")

    def set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    # Actions des boutons principales
    def on_clean_clicked(self):
        self.run_in_thread(self.run_clean_step)

    def on_train_clicked(self):
        self.run_in_thread(self.run_train_step)

    def on_run_all_clicked(self):
        self.run_in_thread(self.run_full_pipeline)

    def on_predict_clicked(self):
        self.open_predict_window()

    # Lancement de tâches en thread séparé
    def run_in_thread(self, target):
        thread = threading.Thread(target=target, daemon=True)
        thread.start()

    # Appel des scripts externes
    def run_clean_step(self):
        self.log("\n>>> Étape 1 : Nettoyage des données")
        self.log(f"Commande : python {NETTOYAGE_SCRIPT}")
        self.set_status("Nettoyage du dataset en cours...")
        self._run_script(NETTOYAGE_SCRIPT, step_name="Étape 1 : Nettoyage")

    def run_train_step(self):
        self.log("\n>>> Étape 2 : Entraînement des modèles (baseline)")
        self.log(f"Commande : python {TRAIN_SCRIPT}")
        self.set_status("Entraînement des modèles en cours...")
        self._run_script(TRAIN_SCRIPT, step_name="Étape 2 : Entraînement")

    def run_full_pipeline(self):
        self.log("\n>>> Lancement du pipeline complet")
        self.set_status("Pipeline complet en cours...")

        ok1 = self._run_script(
            NETTOYAGE_SCRIPT, step_name="Étape 1 : Nettoyage"
        )
        if not ok1:
            self.set_status("Pipeline interrompu (erreur étape 1).")
            return

        ok2 = self._run_script(
            TRAIN_SCRIPT, step_name="Étape 2 : Entraînement"
        )
        if not ok2:
            self.set_status("Pipeline interrompu (erreur étape 2).")
            return

        self.log("\n[OK] Pipeline complet terminé avec succès")
        self.set_status("Pipeline terminé")

    def _run_script(self, script_path, step_name="Étape"):
        if not os.path.exists(script_path):
            self.log(f"[ERREUR] Script introuvable : {script_path}")
            return False

        cmd = [sys.executable, str(script_path)]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(BASE_DIR),
            )

            if result.stdout:
                self.log(result.stdout.strip())

            if result.stderr:
                self.log("=== STDERR ===")
                self.log(result.stderr.strip())

            if result.returncode == 0:
                self.log(f"[OK] {step_name} terminé avec succès")
            else:
                self.log(
                    f"[ERREUR] {step_name} terminé avec code "
                    f"{result.returncode}"
                )
            return result.returncode == 0

        except Exception as e:
            self.log(f"[EXCEPTION] {step_name} : {e}")
            return False
        finally:
            self.set_status("Prêt")

    # Chargement du modèle dans la GUI
    def load_model_if_needed(self):
        if self.model is not None:
            return True
        try:
            self.log(f"Chargement du modèle depuis {BEST_MODEL_PATH} ...")
            self.model = load_model()
            self.log("Modèle chargé en mémoire.")
            return True
        except FileNotFoundError as e:
            msg = str(e)
            self.log("[ERREUR] " + msg)
            messagebox.showerror("Modèle introuvable", msg)
            return False
        except Exception as e:
            self.log(f"[ERREUR] Impossible de charger le modèle : {e}")
            messagebox.showerror(
                "Erreur", f"Impossible de charger le modèle : {e}"
            )
            return False

    # Chargement des valeurs de combobox
    def load_combo_values(self):
        if self.combo_values is not None:
            return self.combo_values

        values = {
            "Country": [
                "United States",
                "United Kingdom",
                "France",
                "Germany",
                "India",
            ],
            "FormalEducation": [
                "Bachelor’s degree (BA, BS, B.Eng., etc.)",
                "Master’s degree (MA, MS, M.Eng., MBA, etc.)",
                "Doctoral degree (Ph.D)",
                "Some college/university study without earning a degree",
            ],
            "UndergradMajor": [
                "Computer science, computer engineering, or software engineering",
                "Information systems, information technology, or system administration",
                "A natural science (ex. biology, chemistry, physics)",
                "Another engineering discipline (ex. civil, electrical, mechanical)",
            ],
            "Employment": [
                "Employed full-time",
                "Employed part-time",
                "Independent contractor, freelancer, or self-employed",
                "Not employed, but looking for work",
            ],
            "DevType": [
                "Full-stack developer",
                "Back-end developer",
                "Front-end developer",
                "Data or business analyst",
            ],
        }

        if CLEANED_DATA_PATH.exists():
            try:
                df = pd.read_csv(
                    CLEANED_DATA_PATH,
                    usecols=[
                        "Country",
                        "FormalEducation",
                        "UndergradMajor",
                        "Employment",
                        "DevType",
                    ],
                )
                for col in values.keys():
                    if col in df.columns:
                        uniques = (
                            df[col]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                        uniques = sorted(set(uniques))
                        if uniques:
                            values[col] = uniques
                self.log(
                    "Valeurs uniques pour les Combobox "
                    "chargées depuis survey_clean.csv."
                )
            except Exception as e:
                self.log(
                    f"[WARN] Impossible de lire survey_clean.csv "
                    f"pour les Combobox : {e}"
                )
        else:
            self.log(
                "[INFO] survey_clean.csv non trouvé, "
                "utilisation des valeurs par défaut pour les Combobox."
            )

        self.combo_values = values
        return values

    # Fenêtre de prédiction de salaire
    def open_predict_window(self):
        if not self.load_model_if_needed():
            return

        if (
            self.predict_window is not None
            and tk.Toplevel.winfo_exists(self.predict_window)
        ):
            self.predict_window.lift()
            return

        combo_vals = self.load_combo_values()

        self.predict_window = tk.Toplevel(self)
        self.predict_window.title("Prédire un salaire")
        self.predict_window.geometry("600x500")
        self.predict_window.configure(bg="#1e1e1e")

        frame = ttk.Frame(self.predict_window, padding=15, style="Main.TFrame")
        frame.pack(fill="both", expand=True)

        title = ttk.Label(
            frame,
            text="Formulaire de prédiction de salaire",
            style="Title.TLabel",
        )
        title.grid(row=0, column=0, columnspan=2, pady=(0, 15))

        self.var_country = tk.StringVar()
        self.var_edu = tk.StringVar()
        self.var_major = tk.StringVar()
        self.var_employment = tk.StringVar()
        self.var_devtype = tk.StringVar()
        self.var_langs = tk.StringVar()
        self.var_years = tk.StringVar()
        self.var_years_prof = tk.StringVar()
        self.var_company_size = tk.StringVar()

        def add_combo(label_text, var, values_list, row_idx):
            lbl = ttk.Label(frame, text=label_text, style="Subtitle.TLabel")
            lbl.grid(
                row=row_idx,
                column=0,
                sticky="w",
                pady=3,
                padx=(0, 10),
            )
            cb = ttk.Combobox(
                frame,
                textvariable=var,
                values=values_list,
                state="readonly",
                width=45,
            )
            cb.grid(row=row_idx, column=1, sticky="w", pady=3)
            if values_list:
                cb.current(0)
            return cb

        def add_entry(label_text, var, row_idx, hint=None):
            lbl = ttk.Label(frame, text=label_text, style="Subtitle.TLabel")
            lbl.grid(
                row=row_idx,
                column=0,
                sticky="w",
                pady=3,
                padx=(0, 10),
            )
            entry = ttk.Entry(frame, textvariable=var, width=47)
            entry.grid(row=row_idx, column=1, sticky="w", pady=3)
            if hint:
                entry.insert(0, hint)
            return entry

        r = 1
        add_combo(
            "Country :", self.var_country, combo_vals.get("Country", []), r
        )
        r += 1
        add_combo(
            "FormalEducation :",
            self.var_edu,
            combo_vals.get("FormalEducation", []),
            r,
        )
        r += 1
        add_combo(
            "UndergradMajor :",
            self.var_major,
            combo_vals.get("UndergradMajor", []),
            r,
        )
        r += 1
        add_combo(
            "Employment :",
            self.var_employment,
            combo_vals.get("Employment", []),
            r,
        )
        r += 1
        add_combo(
            "DevType :", self.var_devtype, combo_vals.get("DevType", []), r
        )
        r += 1

        add_entry(
            "Languages (séparés par ;) :",
            self.var_langs,
            r,
            "JavaScript;Python;HTML;CSS",
        )
        r += 1
        add_entry("YearsCoding :", self.var_years, r, "ex: 5")
        r += 1
        add_entry("YearsCodingProf :", self.var_years_prof, r, "ex: 2")
        r += 1
        add_entry(
            "CompanySize (approx.) :", self.var_company_size, r, "ex: 50"
        )
        r += 1

        self.pred_result_var = tk.StringVar(
            value="Salaire prédit : (en attente)"
        )
        lbl_result = ttk.Label(
            frame,
            textvariable=self.pred_result_var,
            style="Subtitle.TLabel",
        )
        lbl_result.grid(row=r, column=0, columnspan=2, pady=(15, 10))
        r += 1

        btn_predict = ttk.Button(
            frame,
            text="Prédire le salaire",
            style="Primary.TButton",
            command=self.do_predict,
        )
        btn_predict.grid(row=r, column=0, columnspan=2, pady=(5, 0))

    # Logique de prédiction
    def do_predict(self):
        if not self.load_model_if_needed():
            return

        def to_float_or_nan(s):
            s = s.strip()
            if s == "":
                return np.nan
            try:
                return float(s.replace(",", "."))
            except ValueError:
                return np.nan

        country = self.var_country.get().strip()
        edu = self.var_edu.get().strip()
        major = self.var_major.get().strip()
        employment = self.var_employment.get().strip()
        devtype = self.var_devtype.get().strip()
        langs = self.var_langs.get().strip()

        years = to_float_or_nan(self.var_years.get())
        years_prof = to_float_or_nan(self.var_years_prof.get())
        company_size = to_float_or_nan(self.var_company_size.get())

        data = {
            "Country": [country],
            "FormalEducation": [edu],
            "UndergradMajor": [major],
            "YearsCoding": [years],
            "YearsCodingProf": [years_prof],
            "Employment": [employment],
            "CompanySize": [company_size],
            "DevType": [devtype],
            "LanguageWorkedWith": [langs],
        }

        df_input = pd.DataFrame(data)

        try:
            y_pred = self.model.predict(df_input)[0]
            salaire_str = f"{y_pred:,.2f}"
            self.pred_result_var.set(f"Salaire prédit : {salaire_str}")
            self.log(
                f"Prédiction effectuée pour l'input utilisateur → {salaire_str}"
            )
        except Exception as e:
            self.log(f"[ERREUR] Prédiction impossible : {e}")
            messagebox.showerror(
                "Erreur de prédiction", f"Impossible de prédire : {e}"
            )

    # Chargement des données pour les graphes
    def _load_graph_data(self):
        if self.graph_df is not None:
            return True

        try:
            df = load_clean_dataset()
        except Exception as e:
            self.log(f"[ERREUR] Chargement des données pour les graphes : {e}")
            messagebox.showerror(
                "Erreur", f"Impossible de charger les données : {e}"
            )
            return False

        self.graph_df = df
        return True

    # Mise à jour de la figure selon le choix
    def update_graphs(self):
        self._show_graphs_tab()

        if not self._load_graph_data():
            return

        df = self.graph_df.copy()
        sal = df[TARGET].dropna()
        if sal.empty:
            self.log("[WARN] Aucun salaire trouvé pour afficher les graphiques.")
            return

        choix = self.var_graph_type.get()
        self.figure.clear()

        if choix == "Distribution des salaires (global)":
            self._plot_salary_distribution(df)
        elif choix == "Salaire moyen par pays (Top 10)":
            self._plot_salary_by_country(df)
        elif choix == "Salaire moyen par DevType (Top 10)":
            self._plot_salary_by_devtype(df)
        elif choix == "Salaire moyen vs YearsCoding":
            self._plot_salary_vs_years(
                df, col="YearsCoding", title="Salaire moyen vs YearsCoding"
            )
        elif choix == "Salaire moyen vs YearsCodingProf":
            self._plot_salary_vs_years(
                df,
                col="YearsCodingProf",
                title="Salaire moyen vs YearsCodingProf",
            )
        elif choix == "Salaire moyen par niveau d’étude":
            self._plot_salary_by_education(df)
        elif choix == "Performance modèle : Réel vs prédictions (test)":
            self._plot_model_performance(mode="pred_vs_true")
        elif choix == "Performance modèle : Distribution des résidus (test)":
            self._plot_model_performance(mode="residuals_hist")
        elif choix == "Performance modèle : Résidus vs prédictions (test)":
            self._plot_model_performance(mode="residuals_vs_pred")
        elif choix == "Performance modèle : Importance des variables (Top 20)":
            self._plot_model_performance(mode="feature_importance")
        else:
            self._plot_salary_distribution(df)

        self.figure.tight_layout()
        self.canvas.draw()
        self.log(f"Graphique mis à jour : {choix}")

    # Graphique de distribution des salaires
    def _plot_salary_distribution(self, df):
        sal = df[TARGET].dropna()
        high = sal.quantile(0.99)
        sal_clip = sal[sal <= high]

        ax1 = self.figure.add_subplot(1, 2, 1)
        ax1.hist(sal_clip, bins=30)
        ax1.set_title("Distribution des salaires (99% quantile)")
        ax1.set_xlabel("Salaire")
        ax1.set_ylabel("Fréquence")

        ax2 = self.figure.add_subplot(1, 2, 2)
        ax2.boxplot(sal_clip, vert=True, showfliers=False)
        ax2.set_title("Boxplot des salaires")
        ax2.set_xticks([])

    # Graphique salaire moyen par pays
    def _plot_salary_by_country(self, df, top_n=10):
        if "Country" not in df.columns:
            self.log("[WARN] Colonne 'Country' manquante.")
            return

        sub = df[["Country", TARGET]].dropna()
        if sub.empty:
            self.log("[WARN] Pas de données Country + salaire.")
            return

        grp = (
            sub.groupby("Country")[TARGET]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
        )

        ax = self.figure.add_subplot(1, 1, 1)
        ax.bar(grp.index, grp.values)
        ax.set_title(f"Salaire moyen par pays (Top {top_n})")
        ax.set_ylabel("Salaire moyen")
        ax.set_xticklabels(grp.index, rotation=45, ha="right")

    # Graphique salaire moyen par DevType
    def _plot_salary_by_devtype(self, df, top_n=10):
        if "DevType" not in df.columns:
            self.log("[WARN] Colonne 'DevType' manquante.")
            return

        rows = []
        for _, row in df[["DevType", TARGET]].dropna().iterrows():
            devs = str(row["DevType"]).split(";")
            for d in devs:
                d = d.strip()
                if d:
                    rows.append((d, row[TARGET]))

        if not rows:
            self.log("[WARN] Impossible de parser DevType.")
            return

        dev_df = pd.DataFrame(rows, columns=["DevType", TARGET])
        grp = (
            dev_df.groupby("DevType")[TARGET]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
        )

        ax = self.figure.add_subplot(1, 1, 1)
        ax.bar(grp.index, grp.values)
        ax.set_title(f"Salaire moyen par DevType (Top {top_n})")
        ax.set_ylabel("Salaire moyen")
        ax.set_xticklabels(grp.index, rotation=45, ha="right")

    # Graphique salaire moyen en fonction des années
    def _plot_salary_vs_years(self, df, col="YearsCoding", title="Salaire moyen vs années"):
        if col not in df.columns:
            self.log(f"[WARN] Colonne '{col}' manquante.")
            return

        sub = df[[col, TARGET]].dropna()
        if sub.empty:
            self.log(f"[WARN] Pas de données pour {col} + salaire.")
            return

        sub[col] = sub[col].astype(float).round().astype(int)
        grp = sub.groupby(col)[TARGET].mean().sort_index()

        ax = self.figure.add_subplot(1, 1, 1)
        ax.plot(grp.index, grp.values, marker="o")
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Salaire moyen")

    # Graphique salaire moyen par niveau d’étude
    def _plot_salary_by_education(self, df):
        if "FormalEducation" not in df.columns:
            self.log("[WARN] Colonne 'FormalEducation' manquante.")
            return

        sub = df[["FormalEducation", TARGET]].dropna()
        if sub.empty:
            self.log("[WARN] Pas de données pour FormalEducation + salaire.")
            return

        grp = (
            sub.groupby("FormalEducation")[TARGET]
            .mean()
            .sort_values(ascending=False)
        )

        ax = self.figure.add_subplot(1, 1, 1)
        ax.bar(grp.index, grp.values)
        ax.set_title("Salaire moyen par niveau d’étude")
        ax.set_ylabel("Salaire moyen")
        ax.set_xticklabels(grp.index, rotation=45, ha="right")

    # Graphiques de performance du modèle
    def _plot_model_performance(self, mode="pred_vs_true"):
        if not self.load_model_if_needed():
            return

        try:
            df = load_clean_dataset()
        except Exception as e:
            self.log(f"[ERREUR] Chargement des données modèle : {e}")
            return

        from sklearn.model_selection import train_test_split

        X = df[FEATURES].copy()
        y = df[TARGET].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        try:
            y_pred = self.model.predict(X_test)
        except Exception as e:
            self.log(
                f"[ERREUR] Prédiction pour les graphes du modèle impossible : {e}"
            )
            return

        y_true = np.array(y_test)
        y_pred = np.array(y_pred)
        resid = y_true - y_pred

        self.figure.clear()
        ax = self.figure.add_subplot(1, 1, 1)

        if mode == "pred_vs_true":
            vmin = float(min(y_true.min(), y_pred.min()))
            vmax = float(max(y_true.max(), y_pred.max()))
            ax.scatter(y_true, y_pred, alpha=0.3)
            ax.plot([vmin, vmax], [vmin, vmax])
            ax.set_xlabel("Salaire réel (test)")
            ax.set_ylabel("Salaire prédit")
            ax.set_title("Meilleur modèle - Réel vs prédictions (test)")

        elif mode == "residuals_hist":
            ax.hist(resid, bins=30)
            ax.set_xlabel("Résidus (y_true - y_pred)")
            ax.set_ylabel("Fréquence")
            ax.set_title("Meilleur modèle - Distribution des résidus (test)")

        elif mode == "residuals_vs_pred":
            ax.scatter(y_pred, resid, alpha=0.3)
            ax.axhline(0)
            ax.set_xlabel("Valeurs prédites")
            ax.set_ylabel("Résidus")
            ax.set_title("Meilleur modèle - Résidus vs prédictions (test)")

        elif mode == "feature_importance":
            try:
                model_step = self.model.named_steps.get("model", None)
                preproc = self.model.named_steps.get("preprocessor", None)
                if (
                    model_step is None
                    or preproc is None
                    or not hasattr(model_step, "feature_importances_")
                ):
                    self.log(
                        "[WARN] Le modèle ne fournit pas "
                        "d'importances de variables."
                    )
                    return

                feat_names = preproc.get_feature_names_out()
                importances = model_step.feature_importances_
                idx = np.argsort(importances)[::-1][:20]
                top_names = feat_names[idx]
                top_importances = importances[idx]

                ax.bar(range(len(top_importances)), top_importances)
                ax.set_xticks(range(len(top_importances)))
                ax.set_xticklabels(top_names, rotation=90)
                ax.set_ylabel("Importance")
                ax.set_title("Meilleur modèle - Top 20 variables")
            except Exception as e:
                self.log(
                    f"[ERREUR] Impossible d'afficher les importances : {e}"
                )
                return

        self.figure.tight_layout()
        self.canvas.draw()
        self.log(f"Graphique modèle mis à jour ({mode}).")
