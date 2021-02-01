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
from array import array
import numpy as np
from itertools import combinations, permutations

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
    mountpt, mountptdist : the nearest mountpoint & dist
    wholepath : list of points forming the complete path between all apoints, in order
    wholedist : total distance of wholepath
    '''
    def __init__(self, score, apoints):
        self.score = score
        self.apoints = apoints
        self.mountpt = None
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
        env = W, H, mountpoints
        for t in range(T):
            score, n = parseline()
            p = parseline()
            p = [(p[i], p[i+1]) for i in range(0, len(p), 2)]   # turn into 2-ples
            task = Task(score, p)
            # compute route & dist of this task's apoints
            apoints = p
            p = apoints[0]
            wholedist = 0
            wholepath = [p]
            for a in apoints:
                path, d = getroute(env, p, a)
                wholedist += len(path) - 1      # why not use d?
                wholepath += path[1:]
                p = a
            task.wholedist = wholedist
            task.wholepath = wholepath
            tasks.append(task)
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

def play(mountpoints, tasks, maxsteps, dist):
    '''
    schedule every task to available arms.
    stepwise iterate through every path,
    handle collisions by re-routing.
    '''
    arms = [Arm(m) for m in mountpoints]
    
def assign_tasks(mountpoints, tasks):
    '''
    permute tasks to mountpoints recursively
    '''
    from itertools import permutations
    tasks = [1,2,3]
    a = np.array(list(permutations(tasks)))
    a[-1,:].reshape((2,))
    np.array([1,2,3]).reshape((2,-1))

def printenv(env, tasks):
    ''' print an ASCII map of the world '''
    W, H, mountpoints = env
    map = [array('b', b'. ' * W) for _ in range(H)]
    for x,y in mountpoints:
        map[y][x*2] = ord(b'm')
    for x,y in [ap for t in tasks for ap in t.apoints]:
        map[y][x*2] = ord(b'A')
    for m in map:
        print(m.tobytes().decode())

def printroute(path, env):
    ''' print an ASCII map of the world and the given path '''
    W, H, mountpoints = env
    map = [array('b', b'. ' * W) for _ in range(H)]
    for x,y in path:
        map[y][x*2] = ord(b'X')
    for x,y in mountpoints:
        map[y][x*2] = ord(b'm')
    for m in map:
        print(m.tobytes().decode())

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
    printenv(env, tasks)

    # find shortest path & dist from every mountpoint to every task
    # build a distance matrix (rows=mountpoints, cols=tasks)
    distmatrix = []
    for m in mountpoints:       # rows of distmatrix
        print('mountpoint', m)
        drow = []
        for t in tasks:         # cols of distmatrix
            print('  task', t.score, 'points:', end='')
            path, d = getroute(env, m, t.apoints[0])
            drow.append(len(path) + t.wholedist)
            print(' path', len(path), '+ wholepath', t.wholedist)
            # printroute(path, env)
        distmatrix.append(drow)
    distmatrix = np.array(distmatrix)
    # print('distmatrix\n', distmatrix, '\n  summed', [sum(i) for i in distmatrix], sep='')
    print('distmatrix\n', distmatrix, '\n  summed', distmatrix.sum(axis=1), sep='')

    # now try every combination of R mountpoints, assign the arms there
    # and play out a complete game of L steps.
    mountpoints = np.array(mountpoints)
    for m in combinations(np.arange(M), R):
        play(mountpoints[m, :], tasks, L, distmatrix[m, :])


    # now choose the best R mountpoints, and assign the arms there.
    # for t in range(len(tasks)):
    #     print(tasks[t].mountpt, 'assigned to task', t)

    # set each arm with the path to its assigned apoints.


if __name__ == '__main__':
    setup() 