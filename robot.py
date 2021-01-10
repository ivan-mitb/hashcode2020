# https://storage.googleapis.com/coding-competitions.appspot.com/HC/2020/hashcode2020_final_round.pdf

# T tasks, each involving 1 or more assembly points that need to be visited in order
# workspace is W by H

# there are R arms and M mountpoints available.

# T tasks, each involving 1 or more
# assembly points that need to be visited in order.
# the tasks have different value (points) -- we want to maximise
# our reward by prioritising the higher-valued tasks.

# max L steps to perform the tasks.

# output file:
# A number of arms needed, and the mountpoint utilised.
# for each arm:
#     Z number of tasks completed (indices in order)
#     K number of instructions (in order)

# https://towardsdatascience.com/the-bellman-equation-59258a0d3fa7
# how to formulate this problem in terms of V- and Q-functions?
# the state of the system: the game board with entity positions

import heapq
import math

class Arm:
    '''
    mountpoint
    current pos
    assigned task / assembly point / path
    '''
    def __init__(self, mountpoint):
        self.mountpoint = mountpoint
        self.pos = mountpoint

    def move(self, m):
        # check destination available
        # retrns True if OK, False if not
        return True

class Task:
    '''
    score
    apoints : list of assembly points
    '''
    def __init__(self, score, apoints):
        self.score = score
        self.apoints = apoints
    def __repr__(self):
        return '<task: score ' + repr(self.score) + ', apoints ' + repr(self.apoints) + '>'

def read_input(filename):
    def parseline():
        return [int(i) for i in f.readline().strip().split()]

    mountpoints = []
    tasks = []
    with open(filename) as f:
        W, H, R, M, T, L = parseline()
        for m in range(M):
            mountpoints.append(tuple(parseline()))
        for t in range(T):
            score, n = parseline()
            p = parseline()
            p = [(p[i], p[i+1]) for i in range(0, len(p), 2)]   # turn into 2-ples
            tasks.append(Task(score, p))
    return W, H, R, M, T, L, mountpoints, tasks

def getroute(env, p1, p2):
    '''
    returns path & dist between p1 and p2.
    uses heapq (a min-heap) to hopefully pick the shortest path.
    '''
    def within(p):
        return p[0] >= 0 and p[0] < W and p[1] >= 0 and p[1] < H

    def dist(p1, p2, metric='euclid'):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        if metric == 'euclid':
            return math.sqrt(dx*dx + dy*dy)
        elif metric == 'manhattan':
            return abs(dx) + abs(dy)

    W, H, obstacles = env
    Q = []
    visited = []
    # print('departed origin', p1, end=' ')
    heapq.heappush(Q, (dist(p1, p2), [], p1))
    while len(Q) > 0:
        _, prev, x = heapq.heappop(Q)
        if x in visited: continue
        visited.append(x)
        if x == p2:
            # print('arrived dest', p2, 'dist', dist(p1, p2))
            # print(prev+[x])
            return prev+[x], dist(p1, p2, 'manhattan')
        # up-right-down-left
        URDL = [(x[0], x[1]+1), (x[0]+1, x[1]),
                (x[0], x[1]-1), (x[0]-1, x[1])]
        for i in range(4):
            if URDL[i] not in obstacles+visited and within(URDL[i]):
                heapq.heappush(Q, (dist(x, p2), prev+[x], URDL[i]))
    # destination not reachable
    return None, None

def step():
    '''
    update positions of arms and task status
    '''

def setup():
    '''
    read input file
    setup A arms at mountpoints (?how to permute these?)
    assign a (nearest?) task to each arm
    for step in range(L):
        for each arm:
            make a move (?how to permute?)
                    to get nearer to the
                    assigned task's next assembly point
            update arms and tasks
    write output file
    '''
    W, H, R, M, T, L, mountpoints, tasks = read_input('test3.txt')
    env = (W, H, mountpoints)   # the environment (game board)
    print(env, tasks)
    # find shortest path & dist from every mountpoint to every task
    for m in mountpoints:
        print('mountpoint', m)
        for t in tasks:
            p = m
            print('  task', t.score, 'points')
            for a in t.apoints:
                path, d = getroute(env, p, a)
                print('    path', len(path)) #, path)
                p = a
    # now choose the best R mountpoints, and assign the arms there.
    # set each arm with the path to its assigned apoints.

if __name__ == '__main__':
    setup()