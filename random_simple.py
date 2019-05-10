import random
from player import Player
import os
import numpy as np
import chess

class Agent1(Player):
    def __init__(self):
        self.board = None
        self.color = None
        self.my_piece_captured_square = None
        self.invalid_last_move = None
        self.badstate = None
        self.piece_tot = None
        self.piece_tracked = None
        self.missing = None
        self.last_sense = None
        self.depth = 2
        self.last_move = None
        self.last_last_move = None
        self.last_3move = None
        self.last_4move = None
        self.olast_move = None
        self.olast_last_move = None
        self.olast_3move = None
        self.olast_4move = None

    def handle_game_start(self, color, board):
        self.board = board
        self.color = color
        self.piece_tot = [0,8,2,2,2,1,1]
        self.piece_tracked = [0,8,2,2,2,1,1]
        self.missing = 0

    def handle_opponent_move_result(self, captured_my_piece, capture_square):
        # if the opponent captured our piece, remove it from our board.
        self.my_piece_captured_square = capture_square
        if captured_my_piece:
            self.board.remove_piece_at(capture_square)

    def choose_sense(self, sense_actions, move_actions, seconds_left):
        # if our piece was just captured, sense where it was captured
        if self.my_piece_captured_square:
            self.last_sense = self.my_piece_captured_square
            return self.my_piece_captured_square


        # otherwise, just randomly choose a sense action, but don't sense on a square where our pieces are located
        for square, piece in self.board.piece_map().items():
            if piece.color == self.color:
                sense_actions.remove(square)
        self.last_sense = random.choice(sense_actions)
        return self.last_sense

    def check_prev(self,square,piece):
        # if peice hasnt moved me do nothing
        last_piece = self.board.piece_at(square)
        if last_piece and piece.piece_type is last_piece.piece_type:
            return
        elif last_piece:
            self.piece_tracked[last_piece.piece_type] -= 1

        if last_piece and last_piece.piece_type is 6:
            self.check_last(square)

        # returns all the posible previous locations for enemy pieces
        piece_locs = self.board.pieces(piece.piece_type, piece.color)
        attackers_locs = self.board.attackers(piece.color,square)

        poss_moved = []

        for pl in piece_locs:
            if pl in attackers_locs:
                poss_moved.append(pl)

        if len(poss_moved) is 1:
            for ps in poss_moved:
                self.board.remove_piece_at(ps)
        elif len(poss_moved) > 1:
            self.remove_closest(poss_moved,square,piece.piece_type)
        else:
            self.remove_closest(piece_locs,square,piece.piece_type)
        return

    def remove_closest(self, pieces_squares, square, piece_type):
        #skip if the list of possible squares is empty
        if not pieces_squares:
            return

        min = 100
        pr = 0
        sr = chess.square_file(square)
        if piece_type is chess.PAWN:
            #print("pawn")
            new_ps = []
            for ps in pieces_squares:
                if chess.square_file(ps) is sr:
                    new_ps.append(ps)
            if new_ps:
                pieces_squares = new_ps


        for ps in pieces_squares:
            dist = chess.square_distance(ps,square)
            if dist < min:
                pr = ps
                min = dist

        self.board.remove_piece_at(pr)
        #print(pr)
        return

    def format_print_board(self):
        print("Hallucinating Board")
        rows = ['8', '7', '6', '5', '4', '3', '2', '1']
        fen = self.board.board_fen()

        fb = "   A   B   C   D   E   F   G   H  "
        fb += rows[0]
        ind = 1
        for f in fen:
            if f == '/':
                fb += '|' + rows[ind]
                ind += 1
            elif f.isnumeric():
                for i in range(int(f)):
                    fb += '|   '
            else:
                fb += '| ' + f + ' '
        fb += '|'

        ind = 0
        for i in range(9):
            for j in range(34):
                print(fb[ind], end='')
                ind += 1
            print('\n', end='')
        print("")


    def handle_sense_result(self, sense_result):
        # delete the old locations of the opponent pieces
        for square, piece in sense_result:
            if piece is not None and piece.color is not self.color:
                if self.piece_tot[piece.piece_type] > self.piece_tracked[piece.piece_type] and self.piece_tracked[piece.piece_type] > 0:
                    self.piece_tracked[piece.piece_type]+= 1
                else:
                    self.check_prev(square,piece)
        # add the pieces in the sense result to our board
        for square, piece in sense_result:
            if piece is None and self.board.piece_at(square) is not None:
                self.check_last(square)
            self.board.set_piece_at(square, piece)


        #self.format_print_board()
        #print(self.piece_tot)
        #print(self.piece_tracked)

    def check_last(self,square):
        last_piece = self.board.piece_at(square)
        self.piece_tracked[last_piece.piece_type] -= 1
        if last_piece.piece_type is 6:
            k_poss = chess.SQUARES
            """
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            print("lost King")
            """

            min = 100
            pr = 0
            for kp in k_poss:
                #print(self.board.piece_at(kp))
                if self.board.piece_at(kp) is not None or kp is square:
                    continue
                #print(kp)
                dist = chess.square_distance(kp,square)
                if dist < min:
                    pr = kp
                    min = dist
            #print(pr)
            self.board.set_piece_at(pr, last_piece)
            self.piece_tracked[last_piece.piece_type] += 1
        return

    def predict_move(self, seconds_left):
        move_actions = generate_moves(self.board, not self.color)

        cur_board = self.board.copy()
        best_value = float("-inf")
        best_move = None
        if self.olast_last_move is not None and self.olast_last_move in move_actions:
            move_actions.remove(self.olast_last_move)
            if self.olast_4move is not None and self.olast_4move in move_actions:
                move_actions.remove(self.olast_4move)

        if self.olast_move is not None and self.olast_move in move_actions:
            move_actions.remove(self.olast_move)

        for move in move_actions:
            cur_board.push(move)
            value = minimax(self.depth-1, cur_board, float("-inf"), float("inf"), False, not self.color)
            cur_board.pop()

            if value > best_value:
                best_value= value
                best_move = move
        self.olast_4move = self.olast_3move
        self.olast_3move = self.olast_last_move
        self.olast_last_move = self.olast_move
        self.olast_move = best_move
        return best_move

    def choose_move(self, move_actions, seconds_left):
        # if we might be able to take the king, try to
        enemy_king_square = self.board.king(not self.color)
        if enemy_king_square:
            # if there are any ally pieces that can take king, execute one of those moves
            enemy_king_attackers = self.board.attackers(self.color, enemy_king_square)
            if enemy_king_attackers:
                attacker_square = enemy_king_attackers.pop()
                return chess.Move(attacker_square, enemy_king_square)

        move_actions = generate_moves(self.board, self.color)
        cur_board = self.board.copy()
        best_value = float("-inf")
        best_move = None

        if self.last_last_move is not None and self.last_last_move in move_actions:
            move_actions.remove(self.last_last_move)
            if self.last_4move is not None and self.last_4move in move_actions:
                move_actions.remove(self.last_4move)

        if self.last_move is not None and self.last_move in move_actions:
            move_actions.remove(self.last_move)

        for move in move_actions:
            cur_board.push(move)
            value = minimax(self.depth-1, cur_board, float("-inf"), float("inf"), False, self.color)
            cur_board.pop()

            if value > best_value:
                best_value= value
                best_move = move

        return best_move

    def handle_move_result(self, requested_move, taken_move, reason, captured_opponent_piece, capture_square):
        # if a move was executed, apply it to our board
        if taken_move is not None:
            self.board.push(taken_move)
            self.invalid_last_move = None
        else:
            if requested_move is not None:
                self.invalid_last_move = requested_move

        if captured_opponent_piece:
            p = self.board.piece_at(capture_square)
            if p:
                self.piece_tot[p.piece_type] -= 1
                self.piece_tracked[p.piece_type] -= 1
            else:
                self.missing += 1
        self.last_4move = self.last_3move
        self.last_3move = self.last_last_move
        self.last_last_move = self.last_move
        self.last_move = requested_move


    def handle_game_end(self, winner_color, win_reason):
        pass

def generate_moves(board, color):
    board.turn = color
    moves = list(board.generate_pseudo_legal_moves())
    return moves

def minimax(depth, board, alpha, beta, is_max_player, color):
    cur_board = board.copy()

    if depth == 0:
        if color:
            return evaluate(cur_board)
        else:
            return -evaluate(cur_board)

    if is_max_player:
        moves = generate_moves(board, color)
    else:
        moves = generate_moves(board, not color)

    if is_max_player:
        best_value = float("-inf")
        for move in moves:
            cur_board.push(move)
            best_value = max(best_value, minimax(depth - 1, cur_board, alpha, beta, not is_max_player, color))
            cur_board.pop()
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break
        return best_value
    else:
        best_value = float("inf")
        for move in moves:
            cur_board.push(move)
            best_value = min(best_value, minimax(depth - 1, cur_board, alpha, beta, not is_max_player, color))
            cur_board.pop()
            beta = min(beta, best_value)
            if beta <= alpha:
                break
        return best_value

def evaluate(board):
    # board object
    score = 0
    for i, r in enumerate(np.arange(56, -1, -8)):
        for j, c in enumerate(range(8)):
            score += get_piece_value(board.piece_at(r+c), i, j)
    return score

def get_piece_value(piece, i, j):
    if piece is None:
        return 0
    return value_dict[piece.symbol()][i, j]

pawn_value_white = np.array([
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0],
    [1.0,  1.0,  2.0,  3.0,  3.0,  2.0,  1.0,  1.0],
    [0.5,  0.5,  1.0,  2.5,  2.5,  1.0,  0.5,  0.5],
    [0.0,  0.0,  0.0,  2.0,  2.0,  0.0,  0.0,  0.0],
    [0.5, -0.5, -1.0,  0.0,  0.0, -1.0, -0.5,  0.5],
    [0.5,  1.0, 1.0,  -2.0, -2.0,  1.0,  1.0,  0.5],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]]) + 10

knight_value_white = np.array([
    [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
    [-4.0, -2.0,  0.0,  0.0,  0.0,  0.0, -2.0, -4.0],
    [-3.0,  0.0,  1.0,  1.5,  1.5,  1.0,  0.0, -3.0],
    [-3.0,  0.5,  1.5,  2.0,  2.0,  1.5,  0.5, -3.0],
    [-3.0,  0.0,  1.5,  2.0,  2.0,  1.5,  0.0, -3.0],
    [-3.0,  0.5,  1.0,  1.5,  1.5,  1.0,  0.5, -3.0],
    [-4.0, -2.0,  0.0,  0.5,  0.5,  0.0, -2.0, -4.0],
    [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0]]) + 30

bishop_value_white = np.array([
    [ -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
    [ -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
    [ -1.0,  0.0,  0.5,  1.0,  1.0,  0.5,  0.0, -1.0],
    [ -1.0,  0.5,  0.5,  1.0,  1.0,  0.5,  0.5, -1.0],
    [ -1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  0.0, -1.0],
    [ -1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0],
    [ -1.0,  0.5,  0.0,  0.0,  0.0,  0.0,  0.5, -1.0],
    [ -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0]]) + 30

rook_value_white = np.array([
    [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [  0.5,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.5],
    [ -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
    [ -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
    [ -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
    [ -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
    [ -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
    [  0.0,   0.0, 0.0,  0.5,  0.5,  0.0,  0.0,  0.0]]) + 50

queen_value_white = np.array([
    [ -2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
    [ -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
    [ -1.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0],
    [ -0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
    [  0.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
    [ -1.0,  0.5,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0],
    [ -1.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.0, -1.0],
    [ -2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]]) + 90

king_value_white = np.array([
    [ -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
    [ -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
    [ -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
    [ -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
    [ -2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0],
    [ -1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0],
    [  2.0,  2.0,  0.0,  0.0,  0.0,  0.0,  2.0,  2.0 ],
    [  2.0,  3.0,  1.0,  0.0,  0.0,  1.0,  3.0,  2.0 ]]) + 900

pawn_value_black = -pawn_value_white[::-1]
king_value_black = -king_value_white[::-1]
rook_value_black = -rook_value_white[::-1]
bishop_value_black = -bishop_value_white[::-1]
knight_value_black = -knight_value_white
queen_value_black = -queen_value_white

value_dict = {
    "P": pawn_value_white,
    "p": pawn_value_black,
    "N": knight_value_white,
    "n": knight_value_black,
    "B": bishop_value_white,
    "b": bishop_value_black,
    "Q": queen_value_white,
    "q": queen_value_black,
    "K": king_value_white,
    "k": king_value_black,
    "R": rook_value_white,
    "r": rook_value_black
}
