#This is a proof of concept ising solver for transit points

import numpy as np
from scipy.spatial.distance import euclidean

def ising_solver_transit(indir, outdir):
    # Read the data
    # Open the file and read the lines
    with open(indir, "r") as file:
        lines = file.readlines()

    # Read the number of points from the first line
    num_points = int(lines[0].strip())

    # Read the x-y coordinates from the remaining lines
    coordinates = []
    quadrants = []
    for line in lines[1:]:
        x, y, z = map(float, line.strip().split())  # Convert each line to x and y float valuess
        coordinates.append([x, y])
        quadrants.append(z)

    # Group the coordinates into pairs of twos
    # It is guaranteed that the number of points is even
    coordinate_pairs = [[coordinates[i], coordinates[i + 1]] for i in range(0, len(coordinates), 2)]
    print(coordinate_pairs)
    
    candidate_map = {}
    
    for i, pair in enumerate(coordinate_pairs):
        for coord in pair:
            candidates = []
            coord_quadrant = quadrants[coordinates.index(coord)]
            
            for j, other_pair in enumerate(coordinate_pairs):
                # Check first coordinate of pair
                if quadrants[coordinates.index(other_pair[0])] == coord_quadrant:
                    distance = euclidean(coord, other_pair[0])
                    candidates.append((other_pair, distance))
                # Check second coordinate of pair
                elif quadrants[coordinates.index(other_pair[1])] == coord_quadrant:
                    # Reverse the pair so matching coordinate is first
                    reversed_pair = [other_pair[1], other_pair[0]]
                    distance = euclidean(coord, other_pair[1])
                    candidates.append((reversed_pair, distance))
                    
            candidate_map[tuple(coord)] = candidates

    print(candidate_map)