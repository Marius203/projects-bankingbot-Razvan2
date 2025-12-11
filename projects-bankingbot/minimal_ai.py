import tkinter as tk
from tkinter import scrolledtext

RESPONSES = {
    "factura": "Momentan ofer raspunsuri hardcodate, nu stiu nimic:D",
    "salut": "Salut! Eu sunt un AI care ofera raspunsuri hardcodate.",
    "cont": "Contul tau este de la Banca Transilvania.:)",
}

def get_response(message):
    message = message.lower().strip()
    return RESPONSES.get(message, "Nu am inteles intrebarea, incearca din nou.")

def send_message():
    message = user_input.get()
    if message.strip() == "":
        return

    chat_box.insert(tk.END, f"You: {message}\n")

    response = get_response(message)
    chat_box.insert(tk.END, f"AI: {response}\n\n")

    user_input.delete(0, tk.END)
    chat_box.yview(tk.END)

root = tk.Tk()
root.title("Minimal AI Chat")
root.geometry("400x500")

chat_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, state="normal")
chat_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

user_input = tk.Entry(root)
user_input.pack(padx=10, pady=5, fill=tk.X)

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(padx=10, pady=5)

root.bind("<Return>", lambda event: send_message())

root.mainloop()
