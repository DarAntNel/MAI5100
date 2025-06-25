# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        if len(bestIndices) > 1:
            if legalMoves[chosenIndex] == 'Stop':
                bestIndices.pop(chosenIndex)
                chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # # Useful information you can extract from a GameState (pacman.py)

        positive_distance_value = {i: 80 - 2 * i for i in range(31)}
        distance_value = {i: 100 - 20 * i for i in range(4)}
        score = 0
        currentFood = currentGameState.getFood()
        currentCapsules = currentGameState.getCapsules()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        min_distance = min(manhattanDistance(newPos, food) for food in currentFood.asList())

        newGhostStates = successorGameState.getGhostStates()
        for x in newGhostStates:
            m_distance = manhattanDistance(newPos, x.getPosition())
            if m_distance in distance_value and x.scaredTimer < 1:
                score -= distance_value[m_distance]
            if min_distance in positive_distance_value:
                score += positive_distance_value[min_distance]

        return successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, state, depth, agentIndex):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None

        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth - 1 if nextAgent == 0 else depth

        legalMoves = state.getLegalActions(agentIndex)
        if not legalMoves:
            return self.evaluationFunction(state), None

        if agentIndex == 0:
            maxEval = float('-inf')
            bestScoreMove = None
            for move in legalMoves:
                successor = state.generateSuccessor(agentIndex, move)
                evalScore, _ = self.minimax(successor, nextDepth, nextAgent)
                if evalScore > maxEval:
                    maxEval = evalScore
                    bestScoreMove = move
            return maxEval, bestScoreMove

        else:
            minEval = float('inf')
            worseScoreMove = None
            for move in legalMoves:
                successor = state.generateSuccessor(agentIndex, move)
                evalScore, _ = self.minimax(successor, nextDepth, nextAgent)
                if evalScore < minEval:
                    minEval = evalScore
                    worseScoreMove = move
            return minEval, worseScoreMove

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        _, bestAction = self.minimax(gameState, self.depth, self.index)
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def minimaxAB(self, state, depth, agentIndex, alpha, beta):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None

        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth - 1 if nextAgent == 0 else depth

        legalMoves = state.getLegalActions(agentIndex)
        if not legalMoves:
            return self.evaluationFunction(state), None

        if agentIndex == 0:
            maxEval = float('-inf')
            bestScoreMove = None
            for move in legalMoves:
                successor = state.generateSuccessor(agentIndex, move)
                evalScore, _ = self.minimaxAB(successor, nextDepth, nextAgent, alpha, beta)
                if evalScore > maxEval:
                    maxEval = evalScore
                    bestScoreMove = move
                alpha = max(alpha, evalScore)
                if beta < alpha:
                    break
            return maxEval, bestScoreMove

        else:
            minEval = float('inf')
            worseScoreMove = None
            for move in legalMoves:
                successor = state.generateSuccessor(agentIndex, move)
                evalScore, _ = self.minimaxAB(successor, nextDepth, nextAgent, alpha, beta)
                if evalScore < minEval:
                    minEval = evalScore
                    worseScoreMove = move
                beta = min(beta, evalScore)
                if beta < alpha:
                    break
            return minEval, worseScoreMove

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        _, bestAction = self.minimaxAB(gameState, self.depth, self.index, float('-inf'), float('inf'))
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, state, depth, agentIndex):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth - 1 if nextAgent == 0 else depth

        legalMoves = state.getLegalActions(agentIndex)
        if not legalMoves:
            return self.evaluationFunction(state)

        if agentIndex == 0:
            maxEval = float('-inf')
            for move in legalMoves:
                successor = state.generateSuccessor(agentIndex, move)
                evalScore = self.expectimax(successor, nextDepth, nextAgent)
                maxEval = max(maxEval, evalScore)
            return maxEval
        else:
            total = 0
            probability = 1 / len(legalMoves)
            for move in legalMoves:
                successor = state.generateSuccessor(agentIndex, move)
                evalScore = self.expectimax(successor, nextDepth, nextAgent)
                total += probability * evalScore
            return total

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        bestScore = float('-inf')
        bestAction = None
        legalMoves = gameState.getLegalActions(0)

        for action in legalMoves:
            successor = gameState.generateSuccessor(0, action)
            score = self.expectimax(successor, self.depth, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: So I built on my first evaluation function in the reflex agent giving more point to states where pacman in closer to food and less points
    when ghosts are near pacman a combination of these calculations occur on every state.
    In addition to that if the game state is win return infinite positive value and if
    loose an infinite negative value. I also added a bit of code to grant more points to the evaluation when
    the game score is higher.
    """
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    positive_distance_value = {i: 80 - 2 * i for i in range(40)}
    distance_value = {i: 100 - 20 * i for i in range(4)}
    game_score_value = {i: i * 20 for i in range(2001)}

    score = 0
    currentFood = currentGameState.getFood()
    pacmanPosition = currentGameState.getPacmanPosition()
    min_distance = min(manhattanDistance(pacmanPosition, food) for food in currentFood.asList())
    newGhostStates = currentGameState.getGhostStates()

    for x in newGhostStates:
        m_distance = manhattanDistance(pacmanPosition, x.getPosition())
        if m_distance in distance_value and x.scaredTimer < 1:
            score -= distance_value[m_distance]
        if min_distance in positive_distance_value:
            score += positive_distance_value[min_distance]
        if x.scaredTimer > 3:
            score += positive_distance_value[min_distance]

    if currentGameState.getScore() in game_score_value:
        score += game_score_value[currentGameState.getScore()]


    return currentGameState.getScore() + score

# Abbreviation
better = betterEvaluationFunction
