"""Search Algos: MiniMax, AlphaBeta
"""
import time

from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT, np, State
import sys


# TODO: you can import more modules, if needed


class SearchAlgos:
    def __init__(self, utility, succ, perform_move, goal=None):
        """The constructor for all the search algos.
        You can code these functions as you like to, 
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The succesor function.
        :param perform_move: The perform move function.
        :param goal: function that check if you are in a goal state.
        """
        self.utility = utility
        self.succ = succ
        self.perform_move = perform_move

    def search(self, state, depth, maximizing_player):
        pass


class MiniMax(SearchAlgos):
    def __init__(self, utility, succ, perform_move, goal=None):
        super().__init__(utility, succ, perform_move)
        self.goal = goal

    def throw_exception_if_timeout(self, state):
        if state.finish_time - time.time() < 0.3:
            print()
            raise TimeoutError("Yoo")

    def search(self, state, depth, max_player):
        """Start the MiniMax algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param max_player: Whether this is a max node (True) or a min node (False).
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        self.throw_exception_if_timeout(state)
        if self.goal and self.goal(state, max_player): return (self.utility(state, max_player), state.direction)
        if depth == 0: return (self.utility(state, max_player), state.direction)
        childrens = self.succ(state, max_player)
        if max_player:
            currMax = State(None, None, None, None, None, None)
            currMax.value = -np.inf
            for c in childrens:
                v = self.search(c, depth - 1, not max_player)
                c.value = v[0]
                currMax = max(currMax, c)
            return (currMax.value, currMax.direction)
        else:
            currMin = State(None, None, None, None, None, None)
            currMin.value = np.inf
            for c in childrens:
                v = self.search(c, depth - 1, not max_player)
                c.value = v[0]
                currMin = min(currMin, c)
            return (currMin.value, currMin.direction)


class AlphaBeta(SearchAlgos):
    def __init__(self, utility, succ, perform_move, goal=None):
        super().__init__(utility, succ, perform_move)
        self.goal = goal

    def throw_exception_if_timeout(self, state):
        if state.finish_time - time.time() < 0.3:
            print()
            raise TimeoutError("Yoo")

    def search(self, state, depth, max_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param max_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """

        self.throw_exception_if_timeout(state)
        if self.goal and self.goal(state, max_player): return (self.utility(state, max_player), state.direction)
        if depth == 0: return (self.utility(state, max_player), state.direction)
        childrens = self.succ(state, max_player)
        if max_player:
            currMax = State(None, None, None, None, None, None)
            currMax.value = -np.inf
            for c in childrens:
                v = self.search(c, depth - 1, not max_player,alpha,beta)
                c.value = v[0]
                currMax = max(currMax, c)
                alpha = max(currMax.value, alpha)
                if currMax.value >= beta: return np.inf, currMax.direction
            return currMax.value, currMax.direction
        else:
            currMin = State(None, None, None, None, None, None)
            currMin.value = np.inf
            for c in childrens:
                v = self.search(c, depth - 1, not max_player,alpha,beta)
                c.value = v[0]
                currMin = min(currMin, c)
                beta = min(currMin.value, alpha)
                if currMin.value <= alpha: return np.inf, currMin.direction
            return (currMin.value, currMin.direction)
