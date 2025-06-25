# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    visited_states = set()
    actions = util.Stack()
    actions.push((problem.getStartState(), []))

    while actions:
        current_state, path = actions.pop()

        if problem.isGoalState(current_state):
            return path

        if current_state not in visited_states:
            visited_states.add(current_state)

        for successor, action, stepCost in problem.getSuccessors(current_state):
            if successor not in visited_states:
                new_path = path + [action]
                actions.push((successor, new_path))

    return []

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""

    actions = util.Queue()
    actions.push((problem.getStartState(), []))
    visited_states = set()
    visited_states.add(problem.getStartState())

    while actions:
        current_state, path = actions.pop()
        if problem.isGoalState(current_state):
            return path

        for successor, action, stepCost in problem.getSuccessors(current_state):
            if successor not in visited_states:
                visited_states.add(successor)
                actions.push((successor, path + [action]))

    return []

# Modified BFS to work with Q5
# def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
#     """Search the shallowest nodes in the search tree first."""
#     actions = actions = util.Queue()
#     start_state = problem.getStartState()
#     actions.push(((start_state, set()), []))
#
#     visited_states = set()
#     visited_states.add((start_state, frozenset()))
#
#     while actions:
#         (current_position, visited_corners), path = actions.pop()
#
#         if problem.isGoalState((current_position, visited_corners)):
#             return path
#
#         for (next_position, next_visited_corners, action, stepCost) in problem.getSuccessors((current_position, visited_corners)):
#             state_key = (next_position, frozenset(next_visited_corners))
#             if state_key not in visited_states:
#                 visited_states.add(state_key)
#                 actions.push(((next_position, next_visited_corners), path + [action]))
#
#     return []


def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    actions = util.PriorityQueue()
    actions.push((problem.getStartState(), [], 0), 0)
    visited_states = set()

    while actions:
        current_state, path, total_cost = actions.pop()

        if problem.isGoalState(current_state):
            return path

        if current_state not in visited_states:
            visited_states.add(current_state)

            for successor, action, stepCost in problem.getSuccessors(current_state):
                actions.push((successor, path + [action], stepCost + total_cost), stepCost + total_cost)

    return []




def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    actions = util.PriorityQueue()
    actions.push((problem.getStartState(), [], 0), 0)
    visited_states = {}

    while actions:
        current_state, path, total_cost = actions.pop()

        if problem.isGoalState(current_state):
            return path

        if current_state not in visited_states or total_cost < visited_states[current_state]:
            visited_states[current_state] = total_cost

            for successor, action, stepCost in problem.getSuccessors(current_state):
                new_total_cost = total_cost + stepCost
                if successor not in visited_states or new_total_cost < visited_states[successor]:
                    actions.push((successor, path + [action], new_total_cost), new_total_cost + heuristic(successor, problem))

    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
