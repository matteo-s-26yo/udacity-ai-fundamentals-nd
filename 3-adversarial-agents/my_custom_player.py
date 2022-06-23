import random # added here
from sample_players import DataPlayer

class CustomPlayer(DataPlayer):
  """ Implement your own agent to play knight's Isolation

  The get_action() method is the only required method for this project.
  You can modify the interface for get_action by adding named parameters
  with default values, but the function MUST remain compatible with the
  default interface.

  **********************************************************************
  NOTES:
  - The test cases will NOT be run on a machine with GPU access, nor be
  suitable for using any other machine learning techniques.

  - You can pass state forward to your agent on the next turn by assigning
  any pickleable object to the self.context attribute.
  **********************************************************************
  """
  
  def heur(self, state, aggr=1):

    return len(state.liberties(state.locs[self.player_id])) - aggr * len(state.liberties(state.locs[1 - self.player_id]))
    
  def max_value(self, state, depth, alpha, beta):

    if state.terminal_test():
      return state.utility(self.player_id)
    
    if depth<=0:
      return self.heur(state)
      
    v = float("-inf")
    for a in state.actions():
      v = max(v, self.min_value(state.result(a), depth-1, alpha, beta))
      if v >= beta:
        return v
      alpha = max(alpha, v)
    return v

  def min_value(self, state, depth, alpha, beta):

    if state.terminal_test():
      return state.utility(self.player_id)
    
    if depth<=0:
      return self.heur(state)
    
    v = float("inf")
    for a in state.actions():
      v = min(v, self.max_value(state.result(a), depth-1, alpha, beta))
      if v <= alpha:
        return v
      beta = min(beta, v)
    return v

  def action_space_search(self, state, depth):

    alpha = float("-inf")
    beta = float("inf")
    best_score = float("-inf")
    best_move = None
    for a in state.actions():
      v = self.min_value(state.result(a), depth-1, alpha, beta)
      alpha = max(alpha, v)
      if v >= best_score: # non stretto altrimenti non termina
        best_score = v
        best_move = a
    return best_move


  def get_action(self, state):
    """ Employ an adversarial search technique to choose an action
    available in the current state calls self.queue.put(ACTION) at least

    This method must call self.queue.put(ACTION) at least once, and may
    call it as many times as you want; the caller will be responsible
    for cutting off the function after the search time limit has expired.

    See RandomPlayer and GreedyPlayer in sample_players for more examples.

    **********************************************************************
    NOTE: 
    - The caller is responsible for cutting off search, so calling
    get_action() from your own code will create an infinite loop!
    Refer to (and use!) the Isolation.play() function to run games.
    **********************************************************************
    """
    # TODO: Replace the example implementation below with your own search
    # method by combining techniques from lecture
    #
    # EXAMPLE: choose a random move without any search--this function MUST
    # call self.queue.put(ACTION) at least once before time expires
    # (the timer is automatically managed for you)

    """
    # opening book
    # minimax search framework
    # alpha beta pruning
    # iterative deepening
    # heuristics at horizon
    """

    SWITCH = "BOOK" # RANDOM|..|BOOK
    
    iter_deep = 1
    while(True):
      if SWITCH=="RANDOM" or len(state.actions())<=1:
        self.queue.put(random.choice(state.actions()))
      elif SWITCH=="BOOK" and state.ply_count<4 and state.board in self.data.keys():
        self.queue.put(self.data[state.board])
      else:
        self.queue.put(self.action_space_search(state, iter_deep))
      iter_deep += 1

    # import random
    # self.queue.put(random.choice(state.actions()))

""" 
opening book part from now on

# python my_custom_player.py # per costruire il book

# python run_match.py -r 100 -o RANDOM -p 4
# python run_match.py -r 100 -o GREEDY -p 4
# python run_match.py -r 100 -o MINIMAX -p 4
"""

def heur(state, aggr=1): # replaces simulate() from lectures and clones CustomPlayer.heur()

    return len(state.liberties(state.locs[state.player()])) - aggr * len(state.liberties(state.locs[1 - state.player()]))
    
def build_table(num_rounds, depth):
    # Builds a table that maps from game state -> action
    # by choosing the action that accumulates the most
    # wins for the active player. (Note that this uses
    # raw win counts, which are a poor statistic to
    # estimate the value of an action; better statistics
    # exist.)
    
    from collections import defaultdict, Counter
    
    book = defaultdict(Counter)
    for _ in range(num_rounds):
        # print(_)
        state = Isolation()
        build_tree(state, book, depth)
    return {k: max(v, key=v.get) for k, v in book.items()}


def build_tree(state, book, depth):
    
    if depth<=0 or state.terminal_test():
        return -heur(state)
    action = random.choice(state.actions())
    reward = build_tree(state.result(action), book, depth-1)
    book[state.board][action] += reward
    return -reward

if __name__ == "__main__":

  import pickle
  from isolation import Isolation, DebugState

  SAMPLE = 1000000 # maybe too many
  DEPTH = 5 # almeno 4 richiesta

  book = build_table(SAMPLE, DEPTH)
  with open('data.pickle', 'wb') as file:
          pickle.dump(book, file)

  # assert len(book) > 0, "Your opening book is empty"
  # assert all(isinstance(k, tuple) for k in book), "All the keys should be `hashable`"
  # assert all(isinstance(v, tuple) and len(v) == 2 for v in book.values()), "All the values should be tuples of (x, y) actions"
  # print("Looks like your book worked!")
  
  # print(len(book.keys()))
  state0 = Isolation()
  print(DebugState.from_state(state0))
  state1 = state0.result(book[state0.board])
  print(DebugState.from_state(state1))
  state2 = state1.result(book[state1.board])
  print(DebugState.from_state(state2))
