"""
Player for the competition
"""
from players.AbstractPlayer import AbstractPlayer
import utils
from SearchAlgos import AlphaBeta
import time
import numpy as np
import networkx as nx


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time,
                                penalty_score)  # keep the inheritance of the parent's (AbstractPlayer) __init__()
        # TODO: initialize more fields, if needed, and the wanted algorithm from SearchAlgos.py
        self.graph = None

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        self.board = board
        self.creating_initial_graph(board)
        self.max_fruit_turn = min(len(board), len(board[0]))

    def creating_initial_graph(self, board):
        self.graph = nx.Graph()
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 1:
                    self.pos = (i, j)
                self.graph.add_node((i, j))
                if j > 0:
                    self.graph.add_edge((i, j), (i, j - 1))
                if i > 0:
                    self.graph.add_edge((i, j), (i - 1, j))
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == -1:
                    self.graph.remove_node((i, j))

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifying the Player's movement, chosen from self.directions
        """
        finish_time = time.time() + self.turn_time
        depth = 1
        best_move = (-np.inf, (-1, 0))
        while True:
            for direction in self.directions:
                initial_state = utils.State(self.board, direction, self.pos, self.current_turn,
                                            self.fruits_on_board_dict,
                                            finish_time)
                try:
                    outcome = self.minimax_algo.search(initial_state, depth, True)
                    if outcome[0] > best_move[0]:
                        best_move = outcome
                except TimeoutError:
                    self.board[self.pos[0]][self.pos[1]] = -1
                    self.pos = (self.pos[0] + best_move[1][0], self.pos[1] + best_move[1][1])
                    self.board[self.pos[0]][self.pos[1]] = 1

                    return best_move[1]
            depth += 1
            # print('bigger_depth : {} '.format(depth))

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        self.board[pos[0]][pos[1]] = 2
        # Add here changes to graph, need to update opponent pos.

    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        self.fruits_on_board_dict = fruits_on_board_dict

    ########## helper functions in class ##########
    #TODO: add here helper functions in class, if needed

    ########## helper functions for MiniMax algorithm ##########
    # TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm
    def utility(self, state, max_player):
        enemy_pos = None
        available_squares = 0
        fruit_dist = np.inf
        my_pos = self.pos
        for i, l in enumerate(state.board):
            for j, square in enumerate(l):
                if square == 2:
                    enemy_pos = (i, j)
                if square not in [-1, 1, 2]:
                    available_squares += 1
                    if square != 0:
                        fruit_dist = min(fruit_dist, abs(my_pos[0] - i) + abs(my_pos[1] - j))
        distance_from_opponent = abs(my_pos[0] - enemy_pos[0]) + abs(my_pos[1] - enemy_pos[1])
        time_factor = time.time()
        return 0.01 * (1 / distance_from_opponent) + 0.01 * available_squares + 500 / fruit_dist + 0.01 * time_factor

    def succ(self, state, max_player):
        # Expecting board, returns list of boards.
        lst = []
        state.board[state.pos[0]][state.pos[1]] = -1
        for d in self.directions:
            new_pos = (state.pos[0] + d[0], state.pos[1] + d[1])
            i = new_pos[0]
            j = new_pos[1]
            if 0 <= i < len(self.board) and 0 <= j < len(self.board[0]) and (
                    self.board[i][j] not in [-1, 1, 2]):  # then move is legal
                new_board = np.copy(state.board)
                new_board[i][j] = 1 if max_player else 2
                if state.turn + 1 == self.max_fruit_turn:
                    for pos in self.fruits_on_board_dict.keys():
                        if new_board[pos[0]][pos[1]] not in [-1, 1, 2]:
                            new_board[pos[0]][pos[1]] = 0
                lst.append(
                    utils.State(new_board, d, new_pos, state.turn + 1, self.fruits_on_board_dict, state.finish_time))
        return lst

    ########## helper functions for the search algorithm ##########
    # TODO: add here the utility, succ, and perform_move functions used in AlphaBeta algorithm
    def fruit_util(self, state):
        prize_sum = 0
        weighted_sum = 0
        for pos, val in state.fruits_on_board_dictionary:
            d_i = self.calc_dist_to_pos(pos)
            p_i = self.calc_prize(pos, val.value, state)
            prize_sum += p_i
            weighted_sum += p_i / d_i
        return weighted_sum / prize_sum

    def calc_dist_to_pos(self, pos):
        return len(nx.shortest_path(self.graph, source=self.pos, target=pos))

    def calc_prize(self, pos, prize, state):
        return prize  # todo: find better way to classify better prizes
