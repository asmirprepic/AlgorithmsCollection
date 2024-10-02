import numpy as np
""" Not sure if this is actually valid"""


def heuristic(a,b):
    return abs(a[0]-b[0]) + abs(a[1] -b[1])

def a_star_algorithm(start,goal,grid):
    neighbours = [(0,1),(1,0),(0,-1),(-1,0)]

    open_list = [start]
    came_from = {}

    g_score = {start:0}
    f_score = {start: heuristic(start,goal)}

    while open_list:
        current  = min(open_list,key = lambda x: f_score[x])

        if current == goal:
            path = []

            while current in came_from: 
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            return path
        
        for i,j in neighbours:
            neighbour = (current[0]+i,current[1] + j)
            temp_g_score = g_score[current] + 1

            if 0 <= neighbour[0] < grid.shape[0] and 0 <= neighbour[1] < grid.shape[1]:
                if grid[neighbour[0]][neighbour[1]] == 1:
                    continue

                if neighbour not in g_score or temp_g_score < g_score[neighbour]:
                    came_from[neighbour] = current
                    g_score[neighbour] = temp_g_score
                    f_score[neighbour] = temp_g_score + heuristic(neighbour,goal)
                    if neighbour not in open_list:
                        open_list.append(neighbour)

    return []
