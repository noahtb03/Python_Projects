import random

#Authors: Dev Hackett and Noah Bennett

class SlidingPuzzle:
    
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.width = 0
        self.missing = 0

    def get_width(self):
        self.width = int(input("Enter board width: "))

    def get_missing(self):
        #get a random number in the range of values
        self.missing = random.randint(1, self.width * self.width)

    def create_board(self):
        #get a list of all the numbers, make one a blank space, shuffle the numbers order
        nums = list(range(1, self.width * self.width + 1))
        nums[self.missing - 1] = " "
        random.shuffle(nums)
        
        #create initial board
        for i in range(self.width):
            print(" ___" * self.width)
            row = "| " + " | ".join(str(nums[i * self.width + j]) for j in range(self.width)) + " |"
            print(row)

        print(" ___" * self.width)

        self.initial_state = tuple(nums)
    
    def create_goal(self):
        #create goal board with the missing space
        goal_board = list(range(1, self.width * self.width + 1))
        goal_board[self.missing - 1] = " "

        self.goal_state = tuple(goal_board)

    def get_actions(self, state):
        #get row and column of empty index
        empty_index = state.index(" ")
        row, col = divmod(empty_index, self.width)
        moves = {}
        
        #get possible moves and add them to the list of moves
        if row > 0:
            moves[self.apply_move(state, -self.width, empty_index)] = 1
        if row < self.width - 1:
            moves[self.apply_move(state, self.width, empty_index)] = 1
        if col > 0:
            moves[self.apply_move(state, -1, empty_index)] = 1
        if col < self.width - 1:
            moves[self.apply_move(state, 1, empty_index)] = 1

        return moves

    def apply_move(self, state, move, empty_index):
        #Switch the empty index with the number you are moving
        list_state = list(state)
        new_state = list_state[:]
        new_empty_index = empty_index + move
        new_state[empty_index], new_state[new_empty_index] = new_state[new_empty_index], new_state[empty_index]

        return tuple(new_state)
    
    def get_heuristic(self, node):
        heuristic = 0
        n = self.width

        #Get the manhattan distance heuristic value
        for indx, value in enumerate(node.state):
            exp_indx = 0
            if value is not ' ':
                exp_indx = int(value) - 1
            else:
                exp_indx = self.missing
            row_diff = abs((indx // n) - (exp_indx // n))
            col_diff = abs((indx % n) - (exp_indx % n))
            heuristic += row_diff + col_diff

        return heuristic
    
    def print_final(self, state):
        nums = list(state)

        #print the final goal state
        for i in range(self.width):
            print(" ___" * self.width)
            row = "| " + " | ".join(str(nums[i * self.width + j]) for j in range(self.width)) + " |"
            print(row)

        print(" ___" * self.width)