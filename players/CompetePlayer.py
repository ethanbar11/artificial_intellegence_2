"""
Player for the competition
"""
from players.AbstractPlayer import AbstractPlayer
import utils
from SearchAlgos import AlphaBeta
import time
import numpy as np
import networkx as nx
# TODO: TO DELETE
import matplotlib.pyplot as plt


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time,
                                penalty_score)  # keep the inheritance of the parent's (AbstractPlayer) __init__()
        # TODO: initialize more fields, if needed, and the wanted algorithm from SearchAlgos.py
        self.max_fruit_turn = None
        self.penalty_score = penalty_score
        self.directions = utils.get_directions()
        self.game_time = game_time
        self.turn_time = None
        # TODO: Remember update this
        self.current_turn = 0
        self.board = None
        self.pos = None
        self.minimax_algo = AlphaBeta(self.utility, self.succ, None)
        self.opponent_pos = None

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
        self.max_fruit_turn = 2 * min(len(board), len(board[0]))
        self.turn_time = 2 * self.game_time / (len(board) * len(board[0]))

    def creating_initial_graph(self, board):
        self.graph = nx.Graph()
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 1:
                    self.pos = (i, j)
                if board[i][j] == 2:
                    self.opponent_pos = (i, j)
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
        finish_time = time.time() + self.turn_time + 500
        self.current_turn += 1
        depth = 2
        best_move = (-np.inf, (-1, 0))
        while True:

            for direction in self.directions:
                new_pos = (self.pos[0] + direction[0], self.pos[1] + direction[1])
                initial_state = utils.State(self.board, self.graph, direction, new_pos, self.opponent_pos,
                                            self.current_turn,
                                            self.fruits_on_board_dict,
                                            finish_time, None)
                try:
                    outcome = self.minimax_algo.search(initial_state, depth - 1, True)
                    if outcome[0] > best_move[0]:
                        best_move = outcome
                except TimeoutError:
                    self.board[self.pos[0]][self.pos[1]] = -1
                    self.graph.remove_node(self.pos)
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
        self.board[self.opponent_pos[0]][self.opponent_pos[1]] = -1
        self.graph.remove_node(self.opponent_pos)
        self.board[pos[0]][pos[1]] = 2
        self.current_turn += 1
        self.opponent_pos = pos
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
    # TODO: add here helper functions in class, if needed

    ########## helper functions for MiniMax algorithm ##########
    # TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm
    def utility(self, state, is_node_max_player):
        enemy_pos = None
        # available_squares = 0
        # fruit_dist = np.inf
        # my_pos = self.pos
        # for i, l in enumerate(state.board):
        #     for j, square in enumerate(l):
        #         if square == 2:
        #             enemy_pos = (i, j)
        #         if square not in [-1, 1, 2]:
        #             available_squares += 1
        #             if square != 0:
        #                 fruit_dist = min(fruit_dist, abs(my_pos[0] - i) + abs(my_pos[1] - j))
        fruit_util_val = self.fruit_util(state, is_node_max_player=is_node_max_player)
        print('This is fruit util fucker : {} '.format(fruit_util_val))
        # distance_from_opponent = abs(my_pos[0] - enemy_pos[0]) + abs(my_pos[1] - enemy_pos[1])
        # time_factor = time.time()
        # return 0.01 * (1 / distance_from_opponent) + 0.01 * available_squares + 500 / fruit_dist + 0.01 * time_factor
        return fruit_util_val

    def succ(self, state, is_father_max_player):
        # Expecting board, returns list of boards.
        lst = []
        # Applies for opponent and me
        pos_to_remove = state.pos if is_father_max_player else state.opponent_pos
        state.board[pos_to_remove[0]][pos_to_remove[1]] = -1
        state.graph.remove_node(pos_to_remove)

        for direction in self.directions:
            # TODO: Might be mixup between and max player and which player should change posotion.
            my_son_player_pos = state.pos if not is_father_max_player else (
                state.pos[0] + direction[0], state.pos[1] + direction[1])
            opponent_son_pos = (
                state.opponent_pos[0] + direction[0],
                state.opponent_pos[1] + direction[1]) if not is_father_max_player else state.opponent_pos
            changed_son_pos = my_son_player_pos if is_father_max_player else opponent_son_pos
            i = changed_son_pos[0]
            j = changed_son_pos[1]
            if 0 <= i < len(self.board) and 0 <= j < len(self.board[0]) and (
                    self.board[i][j] not in [-1, 1, 2]):  # then move is legal
                new_board = np.copy(state.board)
                new_board[i][j] = 2 if is_father_max_player else 1
                if state.turn + 1 == self.max_fruit_turn:
                    for pos in self.fruits_on_board_dict.keys():
                        if new_board[pos[0]][pos[1]] not in [-1, 1, 2]:
                            new_board[pos[0]][pos[1]] = 0
                fruits_on_board_real_dict = self.fruits_on_board_dict if state.turn + 1 < self.max_fruit_turn else {}
                new_son_state = utils.State(new_board, state.graph, direction, my_son_player_pos, opponent_son_pos,
                                            state.turn + 1,
                                            fruits_on_board_real_dict,
                                            state.finish_time, state.pos)
                lst.append(new_son_state)
        return lst

    ########## helper functions for the search algorithm ##########
    def fruit_util(self, state, is_node_max_player):
        prize_sum = 0
        weighted_sum = 0
        for fruit_pos, val in state.fruits_on_board_dictionary.items():
            d_i = self.calc_dist_to_pos(state.pos if is_node_max_player else state.opponent_pos, fruit_pos)
            if d_i * 2 < self.max_fruit_turn - state.turn:
                p_i = self.calc_prize(fruit_pos, val, state)
                prize_sum += p_i
                weighted_sum += p_i / d_i
        return 0 if prize_sum == 0 else weighted_sum / prize_sum

    def calc_dist_to_pos(self, my_pos, fruit_pos):
        return len(nx.shortest_path(self.graph, source=my_pos, target=fruit_pos)) - 1

    def calc_prize(self, pos, prize, state):
        return prize  # todo: find better way to classify better prizes
