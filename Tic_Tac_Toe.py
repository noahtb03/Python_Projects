import sys


# Displays the board with position numbers (1-9) and the current game state
def print_board(board):
    print()
    print(f" 1 | 2 | 3")
    print("---+---+---")
    print(f" 4 | 5 | 6")
    print("---+---+---")
    print(f" 7 | 8 | 9")
    print()
    print("Current board:")
    print()

    for i, row in enumerate(board):
        print(f" {row[0]} | {row[1]} | {row[2]} ")
        if i < 2:
            print("---+---+---")
    print()


# Gets the current player's move input, validates it against used positions, and returns the row/col and position number
def get_player_move(player, board, player_x_coords, player_o_coords):
    all_coords = player_x_coords + player_o_coords
    while True:
        try:
            position = int(input(f"Player {player}, enter position (1-9): "))
            
            if position < 1 or position > 9:
                print("Please enter a number from 1 to 9.")
                continue
            
            if position in all_coords:
                print("That position is already taken. Try another one.")
                continue
            
            # Convert position (1-9) to row/col
            pos_index = position - 1
            row = pos_index // 3
            col = pos_index % 3
            
            return (row, col), position
        except ValueError:
            print("Please enter a valid number.")


# Toggles the current player between X and O
def switch_player(player):
    return "O" if player == "X" else "X"


# Checks if the game is over by looking for a winner or checking if the board is full
def game_over(board):
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] != " ":
            return True
    
    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != " ":
            return True
    
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != " ":
        return True
    if board[0][2] == board[1][1] == board[2][0] != " ":
        return True
    
    # Check if board is full
    for row in board:
        for cell in row:
            if cell == " ":
                return False
    
    return True


# Returns the winning player (X or O) if there is one, otherwise returns None
def get_winner(board):
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] != " ":
            return row[0]
    
    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != " ":
            return board[0][col]
    
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != " ":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != " ":
        return board[0][2]
    
    return None


# Main game loop that initializes the board and players, then runs the game until someone wins or it's a draw
def main():
    board = [
        [" ", " ", " "],
        [" ", " ", " "],
        [" ", " ", " "]
    ]
    
    player_x_coords = []
    player_o_coords = []
    current_player = "X"
    
    print("Welcome to Tic Tac Toe!")
    
    while not game_over(board):
        print_board(board)
        (row, col), position = get_player_move(current_player, board, player_x_coords, player_o_coords)
        board[row][col] = current_player
        
        if current_player == "X":
            player_x_coords.append(position)
        else:
            player_o_coords.append(position)
        
        current_player = switch_player(current_player)
    
    print_board(board)
    winner = get_winner(board)
    if winner:
        print(f"Player {winner} wins!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    main()
