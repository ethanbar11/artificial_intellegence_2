"""
Player for the competition
"""
from players.AbstractPlayer import AbstractPlayer
# TODO: you can import more modules, if needed
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
        self.max_fruit_turn = min(len(board), len(board[0]))

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        # TODO: erase the following line and implement this function. In case you choose not to use this function,
        # use 'pass' instead of the following line.
        raise NotImplementedError

    ########## helper functions in class ##########
    # TODO: add here helper functions in class, if needed

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
        pass  # todo: calc distance between myself and the position

    def calc_prize(self, pos, prize, state):
        return prize  # todo: some way to classify better prizes
