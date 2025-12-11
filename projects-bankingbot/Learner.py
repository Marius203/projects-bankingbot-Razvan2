import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import threading
import sys
import os

# Asiguram ca importam corect configuratiile pentru a citi folderele disponibile
import config
from rag_core import initialize_rag_system

rag_chain = None
current_subject = "ASC"  # Default


def get_ai_response_thread(message):
    global rag_chain
    try:
        if rag_chain is None:
            response = "Eroare: Sistemul RAG nu este initializat."
        else:
            response = rag_chain.invoke(message)
        root.after(0, display_ai_response, response)

    except Exception as e:
        error_message = f"Eroare: {e}"
        root.after(0, display_ai_response, error_message)


def display_ai_response(response):
    chat_box.config(state="normal")
    chat_box.insert(tk.END, f"AI ({current_subject}): {response}\n\n")
    chat_box.yview(tk.END)
    chat_box.config(state="disabled")

    user_input.config(state="normal")
    send_button.config(state="normal")


def send_message():
    message = user_input.get()
    if message.strip() == "":
        return

    chat_box.config(state="normal")
    chat_box.insert(tk.END, f"You: {message}\n")
    chat_box.yview(tk.END)
    chat_box.config(state="disabled")

    user_input.delete(0, tk.END)
    user_input.config(state="disabled")
    send_button.config(state="disabled")

    chat_box.config(state="normal")
    chat_box.insert(tk.END, "AI: Procesez informatia...\n")
    chat_box.yview(tk.END)
    chat_box.config(state="disabled")

    threading.Thread(target=get_ai_response_thread, args=(message,), daemon=True).start()


def change_subject(event):
    """
    Functie apelata cand userul schimba materia din Dropdown.
    """
    global rag_chain, current_subject
    selected = subject_combo.get()

    if selected == current_subject and rag_chain is not None:
        return

    current_subject = selected
    chat_box.config(state="normal")
    chat_box.insert(tk.END, f"\n--- Schimbare materie: {current_subject} ---\nSe reincarca modelul...\n")
    chat_box.config(state="disabled")

    # Reinitializam RAG-ul in background sa nu blocheze interfata
    threading.Thread(target=reload_rag_system, args=(selected,), daemon=True).start()


def reload_rag_system(subject):
    global rag_chain
    rag_chain = initialize_rag_system(subject)

    # Update UI thread safe
    root.after(0, lambda: messagebox.showinfo("Succes", f"Asistentul a trecut pe materia {subject}"))


def get_available_subjects():
    """Detecteaza automat materiile din folderul DOCS"""
    subjects = ["ASC"]  # Default
    if os.path.exists(config.DOCS_ROOT_PATH):
        dirs = [d for d in os.listdir(config.DOCS_ROOT_PATH) if os.path.isdir(os.path.join(config.DOCS_ROOT_PATH, d))]
        if dirs:
            subjects = sorted(dirs)
    return subjects


def main():
    global root, chat_box, user_input, send_button, subject_combo, rag_chain

    root = tk.Tk()
    root.title("Asistent AI Universitar")
    root.geometry("600x650")

    # --- Zona de Selectie Materie ---
    top_frame = tk.Frame(root, bg="#f0f0f0", pady=5)
    top_frame.pack(fill=tk.X)

    tk.Label(top_frame, text="Alege Materia:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)

    subjects = get_available_subjects()
    subject_combo = ttk.Combobox(top_frame, values=subjects, state="readonly", font=("Arial", 10))
    subject_combo.set(subjects[0] if subjects else "ASC")
    subject_combo.pack(side=tk.LEFT, padx=5)
    subject_combo.bind("<<ComboboxSelected>>", change_subject)

    # --- Chat Area ---
    chat_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, state="disabled", font=("Arial", 10))
    chat_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    input_frame = tk.Frame(root)
    input_frame.pack(padx=10, pady=5, fill=tk.X)

    user_input = tk.Entry(input_frame, font=("Arial", 10))
    user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)

    send_button = tk.Button(input_frame, text="Trimite", command=send_message, width=10, bg="#4CAF50", fg="white")
    send_button.pack(side=tk.RIGHT, padx=(5, 0))

    root.bind("<Return>", lambda event: send_message())

    print("Initializare aplicatie...")
    reload_rag_system(subject_combo.get())

    root.mainloop()


if __name__ == "__main__":
    main()