import numpy as np
from random import randint
import csv


def put_val(board, pos, val):
    board[pos[0]][pos[1]] = val


def choose_random_pos(n, m, illegal_positions):
    pos = (randint(0, n - 1), randint(0, m - 1))
    while pos in illegal_positions:
        pos = (randint(0, n - 1), randint(0, m - 1))
    illegal_positions.append(pos)
    return pos


def generate_board(n=30, m=30, p1_pos=None, p2_pos=None, min_fruits_number=0, max_fruits_number=-1, min_fruits_val=200,
                   max_fruits_val=500, fruits_dict=None, x_squares=None, min_x_squares=0, max_x_squares=-1):
    board = np.zeros((n, m), dtype=int).tolist()
    illegal_positions = []
    number_of_fruits = randint(min_fruits_number, max_fruits_number if max_fruits_number != -1 else ((n + m) // 4))
    number_of_x_squares = randint(min_x_squares, max_x_squares if max_x_squares != -1 else ((n + m) // 5))

    if not p1_pos or not (type(p1_pos) == tuple and len(p1_pos) == 2):
        p1_pos = choose_random_pos(n, m, illegal_positions)

    if not p2_pos or not (type(p2_pos) == tuple and len(p2_pos) == 2):
        p2_pos = choose_random_pos(n, m, illegal_positions)

    if not fruits_dict:
        fruits_dict = dict()
        for i in range(number_of_fruits):
            fruits_dict[choose_random_pos(n, m, illegal_positions)] = randint(min_fruits_val, max_fruits_val)

    if not x_squares:
        x_squares = []
        for i in range(number_of_x_squares):
            x_squares.append(choose_random_pos(n, m, illegal_positions))

    put_val(board, p1_pos, 1)
    put_val(board, p2_pos, 2)
    for pos in x_squares:
        put_val(board, pos, -1)
    for pos, val in fruits_dict.items():
        put_val(board, pos, val)
    return board


def write_to_csv_file(file_name='./boards/generic_board.csv', n=30, m=30, p1_pos=None, p2_pos=None, min_fruits_number=0, max_fruits_number=-1,
                      min_fruits_val=200,
                      max_fruits_val=500, fruits_dict=None, x_squares=None, min_x_squares=0, max_x_squares=-1):
    board = generate_board(n, m, p1_pos, p2_pos, min_fruits_number, max_fruits_number, min_fruits_val, max_fruits_val,
                           fruits_dict, x_squares, min_x_squares, max_x_squares)
    f = open(file_name, 'w')
    text = [' '.join([str(j) for j in i]) for i in board]
    text = '\n'.join(text)
    f.write(text)
    f.close()


if __name__ == "__main__":
    write_to_csv_file(n=7, m=7)
