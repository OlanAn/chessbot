import tkinter as tk
import chess
import chess.svg
from PIL import Image, ImageTk
import io
import cairosvg

from backend.chess_engine import get_best_move_with_minimax


class ChessApp:
    def __init__(self, root):
        self.root = root
        self.board = chess.Board()

        # Label pour afficher l'état de la partie
        self.status_label = tk.Label(root, text="Partie en cours", font=("Helvetica", 14))
        self.status_label.pack()

        # Canvas pour afficher l'échiquier
        self.canvas = tk.Canvas(root, width=600, height=600)
        self.canvas.pack()

        # Bouton pour recommencer la partie
        self.reset_button = tk.Button(root, text="Recommencer", command=self.reset_game, font=("Helvetica", 12))
        self.reset_button.pack()

        # Bouton pour annuler le dernier coup
        self.undo_button = tk.Button(root, text="Annuler", command=self.undo_move, font=("Helvetica", 12))
        self.undo_button.pack()

        # Affiche l'échiquier initial
        self.update_board()

        # Gère les clics sur l'échiquier
        self.canvas.bind("<Button-1>", self.on_click)

        # Variables pour suivre les mouvements
        self.selected_square = None

    def update_board(self):
        """Affiche l'échiquier actuel et met à jour l'état."""
        svg_board = chess.svg.board(self.board).encode("utf-8")

        # Convertir le SVG en PNG avec cairosvg
        png_data = io.BytesIO()
        cairosvg.svg2png(bytestring=svg_board, write_to=png_data)
        png_data.seek(0)

        # Charger l'image dans Pillow
        pil_image = Image.open(png_data).resize((600, 600))
        self.board_image = ImageTk.PhotoImage(pil_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.board_image)

        # Met à jour l'état de la partie
        self.update_status()

    def update_status(self):
        """Met à jour l'état de la partie."""
        if self.board.is_checkmate():
            self.status_label.config(text="Échec et mat ! Partie terminée.")
        elif self.board.is_stalemate():
            self.status_label.config(text="Pat ! Partie terminée.")
        elif self.board.is_insufficient_material():
            self.status_label.config(text="Matériel insuffisant ! Partie terminée.")
        elif self.board.is_check():
            self.status_label.config(text="Échec !")
        else:
            self.status_label.config(text="Partie en cours")

    def on_click(self, event):
        """Gestion des clics sur l'échiquier."""
        square_size = 600 // 8
        col = event.x // square_size
        row = event.y // square_size
        square = chess.square(col, 7 - row)

        if self.selected_square is None:
            self.selected_square = square
        else:
            move = chess.Move(self.selected_square, square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.update_board()
                self.bot_play()
            else:
                self.selected_square = None

    def bot_play(self):
        """Le bot joue un coup."""
        if not self.board.is_game_over():
            bot_move = get_best_move_with_minimax(self.board, depth=2)
            self.board.push(bot_move)
            self.update_board()

    def reset_game(self):
        """Réinitialise l'échiquier."""
        self.board.reset()
        self.update_board()
        self.status_label.config(text="Partie en cours")

    def undo_move(self):
        """Annule le dernier coup."""
        if len(self.board.move_stack) > 0:
            self.board.pop()
            self.update_board()
            self.status_label.config(text="Dernier coup annulé.")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Chess Bot")
    app = ChessApp(root)
    root.mainloop()
