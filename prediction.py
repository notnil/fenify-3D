from __future__ import annotations
from enum import Enum
from pathlib import Path
import chess
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.v2 as T


class Board:
    def __init__(self, board):
        self.board = board

    @classmethod
    def from_file_name(cls, file_name):
        fen = Path(file_name).stem
        fen = fen.replace("-", "/") + " w KQkq - 0 1"
        board = chess.Board(fen=fen)
        return Board(board)

    @classmethod
    def from_array(cls, a):
        board = chess.Board()
        board.clear()
        for file in range(8):
            for rank in range(8):
                i = a[rank][file]
                if i == 0:
                    continue
                sq = (rank * 8) + file
                piece = Board.piece_from_int(i)
                board.set_piece_at(sq, piece)
        return Board(board)

    @classmethod
    def from_prediction(cls, y_hat):
        y_hat = torch.argmax(y_hat, dim=2)
        y_hat = torch.reshape(y_hat, (8, 8))
        return Board.from_array(y_hat.cpu().numpy())

    @classmethod
    def piece_to_int(cls, piece):
        if piece is None:
            return 0
        return piece.piece_type if piece.color else piece.piece_type + 6

    @classmethod
    def piece_from_int(cls, i):
        if i == 0:
            return None
        piece_type = ((i - 1) % 6) + 1
        piece_color = chess.BLACK if i > 6 else chess.WHITE
        return chess.Piece(piece_type=piece_type, color=piece_color)

    def to_array(self):
        a = np.zeros((8, 8), dtype=np.int8)
        for sq, piece in self.board.piece_map().items():
            file = sq % 8
            rank = sq // 8
            a[rank][file] = Board.piece_to_int(piece)
        return a

    def to_prediction(self):
        a = torch.zeros((64, 13), dtype=torch.float)
        for sq, piece in self.board.piece_map().items():
            idx = Board.piece_to_int(piece)
            a[sq][idx] = 1.0
        return a

    def flip(self):
        cp = np.copy(self.to_array())
        flipped = np.fliplr(cp)
        return Board.from_array(flipped)

    def rotate(self, n):
        cp = np.copy(self.to_array())
        rotated = np.rot90(cp, k=4 - n)
        return Board.from_array(rotated)


class PredictionPiece(Enum):
    EMPTY = 0
    WHITE_PAWN = 1
    WHITE_KNIGHT = 2
    WHITE_BISHOP = 3
    WHITE_ROOK = 4
    WHITE_QUEEN = 5
    WHITE_KING = 6
    BLACK_PAWN = 7
    BLACK_KNIGHT = 8
    BLACK_BISHOP = 9
    BLACK_ROOK = 10
    BLACK_QUEEN = 11
    BLACK_KING = 12


class BoardPredictor:
    def __init__(
        self, model_file_path: str = "assets/2022-06-01-chesspic-fen-model-cpu.pt"
    ) -> None:
        self.model = torch.jit.load(model_file_path)
        self.img_transform = T.Compose(
            [
                T.Resize((400, 400)),
                T.Lambda(lambda img: img.convert("RGB")),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, image_file_path: str) -> chess.Board:
        img = Image.open(image_file_path)
        x = self.img_transform(img)
        img_batch = torch.unsqueeze(x, 0)
        y_hat = self.model(img_batch)
        y_hat = torch.squeeze(y_hat)
        y_hat_board = torch.argmax(y_hat, dim=1)
        y_hat_board = torch.reshape(y_hat_board, (8, 8))
        y_hat_board = Board.from_array(y_hat_board.cpu().numpy())
        return y_hat_board.board

    def predict_with_confidence(
        self, image_file_path: str
    ) -> dict[chess.Square, list[float]]:
        img = Image.open(image_file_path)
        x = self.img_transform(img)
        img_batch = torch.unsqueeze(x, 0)
        y_hat = self.model(img_batch)
        y_hat = torch.squeeze(y_hat)
        d = {}
        for sq in range(64):
            d[sq] = y_hat[sq].tolist()
        return d
