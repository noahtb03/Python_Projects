import numpy as np

#Authors: Dev Hackett and Noah Bennett

class Traveling:
  
    # Initializes the instance variables for the Traveling class
    def __init__(self, initial_state, goal_state, graph, sld_map):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.graph = graph
        self.sld_map = sld_map

    @staticmethod
    def build_city_map(file_path: str) -> tuple[dict, dict]:
        # Read the CSV file
        data = np.genfromtxt(file_path, delimiter=',', dtype=str)
        
        # Extract headers (city names) starting after 'City' and 'SLD'
        headers = data[0, 2:]
        city_map = {}
        sld_map = {}

        # Build the nested map structure
        for row in data[1:]:
            city = row[0]
            try:
                # Extract the straight-line distance (SLD) to the goal state
                sld = float(row[1]) if row[1] else None
            except ValueError:
                sld = None

            neighbors = {}
            for i, distance in enumerate(row[2:]):
                if distance:  # Check for valid entries
                    try:
                        neighbors[headers[i]] = float(distance)
                    except ValueError:
                        continue

            city_map[city] = neighbors
            sld_map[city] = sld

        return city_map, sld_map

    def get_actions(self, state):
        # Return possible actions (neighboring cities) from the given state (city)
        return self.graph.get(state, {})

    def get_heuristic(self, node):
        # Return the heuristic (SLD) for a given state, defaulting to 0
        return self.sld_map[node.state]



