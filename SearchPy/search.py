import numpy
import heapq
from Traveling import Traveling
from Sliding import SlidingPuzzle

#Authors: Dev Hackett and Noah Bennett

# Initialize the Search class
class Search:
  # Initialize the Node class within the Search class
  class Node:
    
    # Initialize properties of each node as instance variables
    def __init__(self, state, parent, actions, path_cost, heuristic):
      self.state = state
      self.parent = parent
      self.actions = actions
      self.path_cost = path_cost
      self.heuristic = heuristic
    
    # Less than function used to compare path costs of Nodes without throwing a type error for comparing nodes
    def __lt__(self, other_node):
      return self.path_cost < other_node.path_cost
    
    # Retraces the optimal path to the goal state
    def retrace_path(self):
      current = self
      path = []

      #retrace path until you get to the
      while current.parent is not None:
        path.append(current.state)
        current = current.parent
      
      return path
  
  # Instance variables for the Search Class
  def __init__(self, problem):
    self.problem = problem
    self.solvable = True
  
  # Search algorithim that uses both UCS and A* search depending if the boolean UCS is set equal to true or false.
  def search(self, UCS):
    # Initalize the nstarting node with the initial state of the problem
    node = self.Node(self.problem.initial_state, None, None, 0, None)
    # Initialize the frontier and explored set then pushes the initial node on to the froniter using the priority queue package
    frontier = []
    heapq.heappush(frontier, (node.path_cost, node))
    explored = {}
    while True:
      # Checks if the length of the frontier is 0 and if so the problem can no longer be solved. Returns a boolean false to show
      # that the problem is unsolvable.
      if len(frontier) == 0:
        print("Problem Unsolvable")
        self.solvable = False
        break

      # Removes node off of the frontier and data is stored a a tuple where the first value is the cost and the second is the Node class object
      (node.path_cost, node) = heapq.heappop(frontier)
      
      #if the goal state is reached, retrace path, print path, total cost, expansions
      if self.problem.goal_state == node.state:
        print("Goal State Reached!")
        path = node.retrace_path()
        path.append(problem.initial_state)
        print(path[::-1])
        print(f"Total Cost: {node.path_cost}")
        print("Number of expansions: " + str(len(explored)))
        return [path, len(explored)]
      explored[node.state] = node.path_cost

      # For loop iterates through the map of possible actions given the current state of the node
      for action,distance in self.problem.get_actions(node.state).items():
        child = self.Node(action, node, None, node.path_cost + distance, 0)

        #if UCS skip adding the heuristic
        heuristic = 0
        if not UCS:
            heuristic = self.problem.get_heuristic(child)

        #if the node is not explored or in the frontier add it to both
        if child.state not in explored and child not in frontier:
          explored[child.state] = child.path_cost

          heapq.heappush(frontier, (child.path_cost + heuristic, child))

        #if the cost of a node in the explored set is more than the same node in the frontier update the cost
        elif child.state in explored and child.path_cost < explored[child.state]:
          for (path_cost, node) in frontier:
            if node.state == child.state:
              node.path_cost = child.path_cost
              heapq.heapify(frontier)
              break

#Prompt user for algorithm
print("Would you like to use UCS or A*")
print("1: UCS")
print("2: A*")
algorithm = input("")
print("")

#Prompt user for problem
print("Which problem would you like to solve")
print("1: Traveling Problem")
print("2: Sliding Block Problem")
prob = input("")
print("")

#set UCS based on the user input
UCS = False
if algorithm == '1':
  UCS = True

#based on the problem chosen, prompt the user further and run the algorithm
if prob == '1':
  initial_state = input("Enter your starting city: ")
  goal_state = input("Enter a destination city: ")
  path = "MapInfo.csv"
  graph, heursitcs = Traveling.build_city_map(path)
  problem = Traveling(initial_state=initial_state, goal_state=goal_state, graph = graph, sld_map=heursitcs)
  Searcher = Search(problem=problem)
  Searcher.search(UCS = UCS)

#Do sliding problem
else:
  problem = SlidingPuzzle(None, None)
  problem.get_width()
  problem.get_missing()
  problem.create_goal()
  problem.create_board()
  print("")
  Searcher = Search(problem=problem)
  Searcher.search(UCS = UCS)

  #if problem is unsolvable then don't print the final state
  if Searcher.solvable:
    problem.print_final(problem.goal_state)








