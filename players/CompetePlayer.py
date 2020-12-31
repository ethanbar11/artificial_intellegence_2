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
# TODO: This is probably not legal though we'll check
import copy
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
        self.total_fruit_amount = None

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
        self.graph, self.pos, self.opponent_pos = create_graph_of_board(board)

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifying the Player's movement, chosen from self.directions
        """
        finish_time = time.time() + self.turn_time + 500
        depth = 1
        best_move = (-np.inf, (-1, 0))
        initial_state = utils.State(self.board, self.graph, (0, 0), self.pos, self.opponent_pos,
                                    self.current_turn,
                                    self.fruits_on_board_dict,
                                    finish_time, None)
        while True:
            try:
                best_move = self.minimax_algo.search(initial_state, depth, True)
            except TimeoutError:
                self.current_turn += 1
                self.board[self.pos[0]][self.pos[1]] = -1
                self.graph.remove_node(self.pos)
                self.pos = (self.pos[0] + best_move[1][0], self.pos[1] + best_move[1][1])
                self.board[self.pos[0]][self.pos[1]] = 1

                return best_move[1]
            print('depth: {} best move is to move to : {} with grade of : {}/16'.format(depth,
                                                                                        (best_move[1][0] + self.pos[0],
                                                                                         best_move[1][1] + self.pos[1]),
                                                                                        16 * best_move[0]))
            depth += 1
            # print('bigger_depth : {} '.format(depth))

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        if pos in self.fruits_on_board_dict:
            del self.fruits_on_board_dict[pos]
        self.board[self.opponent_pos[0]][self.opponent_pos[1]] = -1
        self.graph.remove_node(self.opponent_pos)
        self.board[pos[0]][pos[1]] = 2
        self.current_turn += 1
        self.opponent_pos = pos

    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        self.total_fruit_amount = 0
        self.fruits_on_board_dict = fruits_on_board_dict
        for pos, val in self.fruits_on_board_dict.items():
            self.total_fruit_amount += val

    ########## helper functions in class ##########
    # TODO: add here helper functions in class, if needed

    ########## helper functions for MiniMax algorithm ##########
    # TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm
    def utility(self, state, is_father_max_node):
        NEW_MAX = 100
        weights = {'fruit_util': 0.5, 'opponent_fruits_util': 0.5}
        fruit_util_val = self.fruit_util(state, True)
        fruit_util_opponent = self.fruit_util(state, False)
        utils_val = \
            weights['fruit_util'] * fruit_util_val + \
            weights['opponent_fruits_util'] * fruit_util_opponent
        # TODO: Change to converted value
        converted_value = (utils_val + 1) * NEW_MAX  # grade value from 0 to 100
        return utils_val

    def succ(self, state, is_father_max_player):
        # Expecting board, returns list of boards.
        successors = []

        for direction in self.directions:
            changed_son_pos, my_son_player_pos, opponent_son_pos = self.calculate_new_poses(direction,
                                                                                            is_father_max_player, state)
            i = changed_son_pos[0]
            j = changed_son_pos[1]
            if self.is_move_legal(state, i, j):  # then move is legal
                self.add_succesor_to_list(changed_son_pos, direction, i, is_father_max_player, j, my_son_player_pos,
                                          opponent_son_pos, state, successors)
        return successors

    def add_succesor_to_list(self, changed_son_pos, direction, i, is_father_max_player, j, my_son_player_pos,
                             opponent_son_pos,
                             state, successors):
        new_board, new_graph, fruits_on_board_real_dict = self.update_graph_and_board(changed_son_pos, i,
                                                                                      is_father_max_player, j, state)
        new_son_state = utils.State(new_board, new_graph, direction, my_son_player_pos, opponent_son_pos,
                                    state.turn + 1,
                                    fruits_on_board_real_dict,
                                    state.finish_time, state.pos)
        successors.append(new_son_state)

    def is_move_legal(self, state, i, j):
        return 0 <= i < len(state.board) and 0 <= j < len(state.board[0]) and (
                state.board[i][j] not in [-1, 1, 2])

    def calculate_new_poses(self, direction, is_father_max_player, state):
        my_son_player_pos = state.pos if is_father_max_player else (
            state.pos[0] + direction[0], state.pos[1] + direction[1])
        opponent_son_pos = (
            state.opponent_pos[0] + direction[0],
            state.opponent_pos[1] + direction[1]) if is_father_max_player else state.opponent_pos
        changed_son_pos = my_son_player_pos if not is_father_max_player else opponent_son_pos
        return changed_son_pos, my_son_player_pos, opponent_son_pos

    def update_graph_and_board(self, changed_son_pos, i, is_father_max_player, j, state):
        pos_to_remove = state.opponent_pos if is_father_max_player else state.pos

        fruits_on_board_real_dict = self.update_fruit_dict(state, changed_son_pos)
        new_graph = self.update_graph(changed_son_pos, is_father_max_player, pos_to_remove, state)
        new_board = self.update_board(i, is_father_max_player, j, pos_to_remove, state)
        return new_board, new_graph, fruits_on_board_real_dict

    def update_fruit_dict(self, state, changed_son_pos):
        fruits_on_board_real_dict = copy.deepcopy(
            state.fruits_on_board_dictionary) if state.turn + 1 < self.max_fruit_turn else {}

        if changed_son_pos in fruits_on_board_real_dict:
            del fruits_on_board_real_dict[changed_son_pos]
        return fruits_on_board_real_dict

    def update_graph(self, changed_son_pos, is_father_max_player, pos_to_remove, state):
        new_graph = state.graph.copy()
        # If moving our player - delete last position
        new_graph.remove_node(state.opponent_pos if is_father_max_player else state.pos)
        # if not is_father_max_player:
        #     new_graph.remove_node(state.pos)
        # else:
        #     new_graph.remove_node(changed_son_pos)
        return new_graph

    def update_board(self, i, is_father_max_player, j, pos_to_remove, state):
        state.board[pos_to_remove[0]][pos_to_remove[1]] = -1

        new_board = np.copy(state.board)
        new_board[i][j] = 2 if is_father_max_player else 1
        if state.turn + 1 == self.max_fruit_turn:
            for pos in self.fruits_on_board_dict.keys():
                if new_board[pos[0]][pos[1]] not in [-1, 1, 2]:
                    new_board[pos[0]][pos[1]] = 0
        return new_board

    ########## helper functions for the search algorithm ##########
    def fruit_util(self, state, my_player_heurisitic):
        g2 = state.graph.copy()
        edges = None
        pos_to_remove = state.opponent_pos if my_player_heurisitic else state.pos
        path_beginning_pos = state.opponent_pos if not my_player_heurisitic else state.pos
        # Deleting from graph the opponent
        edges = [i for i in state.graph.edges(pos_to_remove)]
        state.graph.remove_node(pos_to_remove)
        weighted_sum = 0
        # RETURN HERE
        for fruit_pos, val in state.fruits_on_board_dictionary.items():
            d_i = self.calc_dist_to_pos(state, path_beginning_pos, fruit_pos)
            p_i = self.calc_prize(fruit_pos, val, state)
            if d_i * 2 <= self.max_fruit_turn - state.turn:
                weighted_sum += p_i / d_i
        # Restore graph
        state.graph.add_node(pos_to_remove)
        state.graph.add_edges_from(edges)
        ret_val = 0 if weighted_sum == 0 else weighted_sum / self.total_fruit_amount
        return ret_val if my_player_heurisitic else -ret_val

    def calc_dist_to_pos(self, state, my_pos, fruit_pos):
        if nx.has_path(state.graph, source=my_pos, target=fruit_pos):
            return len(nx.shortest_path(state.graph, source=my_pos, target=fruit_pos)) - 1
        return np.inf

    def calc_prize(self, pos, prize, state):
        return prize  # todo: find better way to classify better prizes

    def curr_score_util(self, state):
        pass


def create_graph_of_board(board):
    global opponent_pos
    graph = nx.Graph()
    pos = None
    opponent_pos = None
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 1:
                pos = (i, j)
            if board[i][j] == 2:
                opponent_pos = (i, j)
            graph.add_node((i, j))
            if j > 0:
                graph.add_edge((i, j), (i, j - 1))
            if i > 0:
                graph.add_edge((i, j), (i - 1, j))
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == -1:
                graph.remove_node((i, j))

    return graph, pos, opponent_pos
