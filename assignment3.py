"""
====================================================+
Assignment 3                                        |
----------------------------------------------------| 
Name:       : Garret Yong Shern Min                 |
Student ID  : 31862616                              |
Unit Code   : FIT2004                               |
====================================================+
"""


"""
This marks the start of Task 1 of Assignment 2
=====|Task 1 Start|==================================================================================
"""
import math
from symbol import term


"""
This class is the class used to represent edges between two vertices 
"""
class Edge:
    def __init__(self,u ,v, capacity, flow):
        # start
        self.u = u
        # end
        self.v = v
        # capacity
        self.capacity = capacity
        # flow
        self.flow = flow

    def __str__(self):
        output_str = str(self.u) + " ," + str(self.v) + " ," + str(self.flow) + "/" + str(self.capacity)
        return output_str

class ResidualEdge:
    def __init__(self,u ,v, flow):
        # start
        self.u = u
        # end
        self.v = v
        # flow
        self.flow = flow

    def __str__(self):
        output_str = str(self.u) + " ," + str(self.v) + " ," + str(self.flow)
        return output_str

"""
This class is used to represent a vertex of a graph
"""
class Vertex:
    def __init__(self, id):
        self.id = id
        self.edges = []  # list of edges

        # Statuses for traversal
        self.discovered = False
        self.visited = False

        # Distance of vertices from source node
        self.distance = 0

        # Backtrack (previous vertex)
        self.previous = None

    def add_edge(self, edge_to_add: Edge):
        self.edges.append(edge_to_add)

    def discovered_vertex(self):
        self.discovered = True
    def reset_discovered(self):
        self.discovered = False

    def visited_vertex(self):
        self.visited = True
    def reset_visited(self):
        self.visited = False

    def reset_discovered_visited(self):
        self.reset_discovered()
        self.reset_visited()

    def __str__(self):
        output_str = str(self.id)
        for edge in self.edges:
            output_str = output_str + "\n with edge(s):" + str(edge)
        return output_str


class Graph:
    """
    This is the constructor for the Graph class

    Time complexity         : O(n^2)
    """
    """
    This flow network graph has 7 sets
    - source
    - friends
    - days (per friend) so n * 5 nodes 
    - meals that one friend can prepare
    - actual meals being prepared
    - sink
    - super sink
    
    """
    def __init__(self, availability, lower_bound, upper_bound, restaurant_bound):
        self.lower_bound = math.floor(lower_bound*len(availability))
        self.upper_bound = math.ceil(upper_bound*len(availability))
        # self.lower_bound = 1
        # self.upper_bound = 2
        self.restaurant_bound = math.ceil( restaurant_bound*len(availability))
        
        self.NUMBER_OF_FRIENDS = 5 # number of friends including me 
        
        # Total number of nodes to represent everything
        max_index = 3 + self.NUMBER_OF_FRIENDS + self.NUMBER_OF_FRIENDS*len(availability)*3 + len(availability)*2  # last +2 is for one source and one sink and one super sink

        # Setting up the number of locations (empty)
        self.vertices = [None] * (max_index)

        # Setting up nodes     
        for i in range(max_index):
            self.vertices[i] = Vertex(i)

        """
        Note to self,
        Index 0 is source
        Index 1 to 5 is friends and you
        Index 5 + 1 to 5 + 5 * len(edge_list) is days
        Index 5 + 5 * len(edge_list) + 1 to 5 + 5 * len(edge_list) * 3 is meals that can be prepared by the friend
        Index 5 + 5 * len(edge_list) * 3 + 1 to 5 + 5 * len(edge_list) * 3 + len(edge_list) * 2 is actual meals 
        Index 5 + 5 * len(edge_list) * 3 + len(edge_list) * 2 + 1 is the sink
        Index 5 + 5 * len(edge_list) * 3 + len(edge_list) * 2 + 2 is the super sink
        """
        # Add corresponding edge to form a flow network 
        self.add_edges(availability)

    """
    This method takes in a list of roads and allocates it based on their start location, u

    Time complexity         : O(n)
    """
    def add_edges(self, availability):
        """
        Note to self,
        Index 0 is source
        Index 1 to 5 is friends and you
        Index 5 + 1 to 5 + 5 * len(edge_list) is days
        Index 5 + 5 * len(edge_list) + 1 to 5 + 5 * len(edge_list) * 3 is meals that can be prepared by the friend
        Index 5 + 5 * len(edge_list) * 3 + 1 to 5 + 5 * len(edge_list) * 3 + len(edge_list) * 2 is actual meals 
        Index 5 + 5 * len(edge_list) * 3 + len(edge_list) * 2 + 1 is the sink
        Index 5 + 5 * len(edge_list) * 3 + len(edge_list) * 2 + 2 is the super sink
        """
        days_counter = self.NUMBER_OF_FRIENDS+1
        for i in range(1, self.NUMBER_OF_FRIENDS+1):
            # Add edges between the source and the friends
            current_edge = Edge(0, i, self.lower_bound, 0)
            self.vertices[0].add_edge(current_edge)

            # Add edges between friends and days
            for j in range(len(availability)):
                current_edge = Edge(i, days_counter, 1, 0)
                self.vertices[i].add_edge(current_edge)
                days_counter += 1
        
        # Link sink to super sink
        current_edge = Edge((5 + 5 * len(availability) * 3 + len(availability) * 2 + 1),(5 + 5 * len(availability) * 3 + len(availability) * 2 + 2),(len(availability)*2),0)
        self.vertices[5 + 5 * len(availability) * 3 + len(availability) * 2 + 1].add_edge(current_edge)

        # Link meals friends can prepare to actual meals prepared
        actual_meals_prepared_counter = 5 + 5 * len(availability) * 3 + 1
        for i in range(1, len(availability)*2+1):
            meals_that_can_be_prepared_counter = 5 + 5 * len(availability) + i
            spaces_between_each_meal = 2 * len(availability)
            for j in range(self.NUMBER_OF_FRIENDS):
                current_edge = Edge(meals_that_can_be_prepared_counter, actual_meals_prepared_counter, 1, 0)
                self.vertices[meals_that_can_be_prepared_counter].add_edge(current_edge)
                meals_that_can_be_prepared_counter += spaces_between_each_meal
            actual_meals_prepared_counter += 1

        # Link actual meals prepared to the sink
        for i in range((5 + 5 * len(availability) * 3 + 1) , (5 + 5 * len(availability) * 3 + len(availability) * 2 + 1)):
            current_edge = Edge(i, (5 + 5 * len(availability) * 3 + len(availability) * 2 + 1), 1, 0)
            self.vertices[i].add_edge(current_edge)
            
        # Link days to meals tha each frind can prepare
        days_counter = self.NUMBER_OF_FRIENDS+1
        meals_that_can_be_prepared_counter = 5 + 5 * len(availability) + 1
        spaces_between_each_days_meal = 2
        for j in range(self.NUMBER_OF_FRIENDS):
            for i in range(len(availability)):
                if availability[i][j] == 1:
                    current_edge = Edge(days_counter, meals_that_can_be_prepared_counter, 1, 0)
                    self.vertices[days_counter].add_edge(current_edge)
                elif availability[i][j] == 2:
                    current_edge = Edge(days_counter, meals_that_can_be_prepared_counter+1, 1, 0)
                    self.vertices[days_counter].add_edge(current_edge)
                elif availability[i][j] == 3:
                    current_edge = Edge(days_counter, meals_that_can_be_prepared_counter, 1, 0)
                    self.vertices[days_counter].add_edge(current_edge)
                    current_edge = Edge(days_counter, meals_that_can_be_prepared_counter+1, 1, 0)
                    self.vertices[days_counter].add_edge(current_edge)
                meals_that_can_be_prepared_counter += spaces_between_each_days_meal
                days_counter += 1
    
    def bfs(self, source, destination):
        """
        Return true if there is path to sink

        tbh, I am not sure if this is used but am too scared to delete
        """
        for i in range(len(self.vertices)):
            self.vertices[i].reset_discovered_visited()
        
        discovered = []
        discovered.append(source)
        while len(discovered) > 0:
            # serve from
            u = self.vertices[discovered.pop(0)]
            u.visited = True
            
            for edge in u.edges:
                v = self.vertices[edge.v] 
                if (v.discovered == False) and ((edge.capacity - edge.flow)>0):
                    discovered.append(v.id)
                    v.discovered = True
                    v.previous = u
                    if v.id == destination:
                        return True
        return False 

    def __str__(self):
        output_str = ""
        for vertex in self.vertices:
            output_str = output_str + "Vertex " + str(vertex) + "\n"
        return output_str
    
class Residual_Graph:
    """
    This is the constructor for the Graph class

    Time complexity         : O(|V | + |E|)
    Aux Space complexity    : O(E)
    """
    def __init__(self, graph):
        
        self.vertices = [None] * (len(graph.vertices))

        for i in range(len(graph.vertices)):
            self.vertices[i] = Vertex(i)

        for vertex in graph.vertices:
            for edge in vertex.edges:
                u = edge.u #start
                v = edge.v #destination
                capacity = edge.capacity
                flow = edge.flow
                forward_edge = ResidualEdge(u,v,(capacity - flow))
                backward_edge = ResidualEdge(v,u,flow)
                self.vertices[u].add_edge(forward_edge)
                self.vertices[v].add_edge(backward_edge)
    
    def bfs(self, source, destination):
        """
        Return true if there is path to sink
        """
        for i in range(len(self.vertices)):
            self.vertices[i].reset_discovered_visited()
        
        discovered = []
        discovered.append(source)
        while len(discovered) > 0:
            # serve from
            u = self.vertices[discovered.pop(0)]
            u.visited = True
            
            for edge in u.edges:
                v = self.vertices[edge.v] 
                if (v.discovered == False) and ((edge.flow)>0):
                    discovered.append(v.id)
                    v.discovered = True
                    v.previous = u
                    if v.id == destination:
                        return True
        return False 

    def __str__(self):
        output_str = ""
        for vertex in self.vertices:
            output_str = output_str + "Vertex " + str(vertex) + "\n"
        return output_str

"""
Ford Fulkerson algorithm to max out the flow based on residual graph information
"""
def ford_fulkerson(graph, source, sink):
    flow = 0

    # Instantiate residual network
    residual_network = Residual_Graph(graph)

    # While there is a path from source to sink (.bfs also manipulates the residual graph to allocate the prev nodes)
    while residual_network.bfs(source, sink):
        path = []
        path.append(residual_network.vertices[sink])
        backtrack = residual_network.vertices[sink].previous
        prev_node_id = sink
        # Instantiate a minimum available flow for use later
        min_available_flow = math.inf
        # Backtracking
        while backtrack is not None:
            path.append(backtrack)
            for edge in backtrack.edges:
                if edge.v == prev_node_id:
                    if (edge.flow) < min_available_flow:
                        min_available_flow = (edge.flow)
            prev_node_id = backtrack.id
            backtrack = backtrack.previous
            # break early if backtrack to source
            if backtrack.id == 0:
                path.append(backtrack)
                break
        
        # Raise error in case so weird error
        if min_available_flow == math.inf:
            raise ValueError("Wut der Hel")
        
        # Augment the flow equal to the residual capacity 
        flow += min_available_flow

        # updating the forward edges  
        prev_node_id = 40
        for vertex in path:
            for edge in vertex.edges:
                if edge.v == prev_node_id:
                    edge.flow -= min_available_flow
            prev_node_id = vertex.id

        # updating the backwards edges
        prev_node_id = 0
        for vertex in path[::-1]:
            for edge in vertex.edges:
                if edge.v == prev_node_id:
                    edge.flow += min_available_flow
            prev_node_id = vertex.id
    
    # Update the actual graph
    for i in range(len(residual_network.vertices)-1,-1,-1):
        for edge in residual_network.vertices[i].edges:
            if edge.v < edge.u: # then it is a node from a the previous set
                for graph_edge in graph.vertices[edge.v].edges:
                    if (graph_edge.u == edge.v) and (graph_edge.v == edge.u):
                        graph_edge.flow = edge.flow
        
    return flow

"""
This method is a method to allocate people to arrage meals

Complexity: O(n^2)
"""
def allocate(availability):
    # Initialize graph
    myGraph = Graph(availability, 0.36, 0.44, 0.1)
    
    # Run Ford Fulkerson to meet lower bound
    ford_fulkerson(myGraph, 0 ,len(myGraph.vertices)-1)

    # Update the relevant edges to prepare for upper bound Ford Fulkerson 
    for edge in myGraph.vertices[0].edges:
        edge.capacity = myGraph.upper_bound
    for i in range(1, myGraph.NUMBER_OF_FRIENDS+1):
        for edge in myGraph.vertices[i].edges:
            edge.capacity = myGraph.upper_bound

    # Run Ford Fulkerson to max out flow while constraining the flow to maximise the flow
    ford_fulkerson(myGraph, 0 ,len(myGraph.vertices)-1)

    # Terminate early and return appropriate value (None) if there is no feasible way to allocate manpower
    output_capacity = myGraph.vertices[len(myGraph.vertices)-2].edges[0].capacity
    output_flow = myGraph.vertices[len(myGraph.vertices)-2].edges[0].flow
    if (output_capacity - output_flow) > myGraph.restaurant_bound:
        return None

    # Setting some variables for calculation later
    number_of_days = len(availability)
    number_of_meal_options_per_person = number_of_days * 2 
    breakfast_allocation = [None] * number_of_days
    dinner_allocation = [None] * number_of_days
    
    # Finding the relavant nodes that have been used to indicate which meal that was prepared by the friend
    # (still no info on [meal_type, person_preparing and day] )
    meal_serving_indexes = []
    for friends in range(1, myGraph.NUMBER_OF_FRIENDS+1):
        for day_edge in myGraph.vertices[friends].edges:
            if day_edge.flow > 0:
                for meal_edge in myGraph.vertices[day_edge.v].edges:
                    if meal_edge.flow > 0:
                        meal_serving_indexes.append(meal_edge.v)

    # Based on the meals that was prepared by friends, find who exactly served which meal
    individuals = []
    for meal in meal_serving_indexes:
        if (meal >= 5 + 5 * len(availability) + 1) and (meal < ((5 + 5 * len(availability) + 1)+number_of_meal_options_per_person)):
            individuals.append(0)
        if (meal >= ((5 + 5 * len(availability) + 1)+number_of_meal_options_per_person)) and (meal < ((5 + 5 * len(availability) + 1)+number_of_meal_options_per_person*2)):
            individuals.append(1)
        if (meal >= ((5 + 5 * len(availability) + 1)+number_of_meal_options_per_person*2)) and (meal < ((5 + 5 * len(availability) + 1)+number_of_meal_options_per_person*3)):
            individuals.append(2)
        if (meal >= ((5 + 5 * len(availability) + 1)+number_of_meal_options_per_person*3)) and (meal < ((5 + 5 * len(availability) + 1)+number_of_meal_options_per_person*4)):
            individuals.append(3)
        if (meal >= ((5 + 5 * len(availability) + 1)+number_of_meal_options_per_person*4)) and (meal < ((5 + 5 * len(availability) + 1)+number_of_meal_options_per_person*5)):
            individuals.append(4)

    # Based on the meals that was prepared by friends, find what type of meal (breakfast or dinner) was prepared for that node 
    breakfast_or_dinner = []
    point_of_reference = (5 + 5 * len(availability) * 3 + 1) %2
    if point_of_reference == 0:
        for meal in meal_serving_indexes:
            if (meal % 2) == 0:
                breakfast_or_dinner.append('b')
            else:
                breakfast_or_dinner.append('d')
    else:
        for meal in meal_serving_indexes:
            if (meal % 2) == 1:
                breakfast_or_dinner.append('b')
            else:
                breakfast_or_dinner.append('d')
            
    # Based on the meals that was prepared by friends, find which day was each meal prepared for
    day_index = []
    first_meal_each_friend_can_serve_index  = 5 + 5 * len(availability) + 1
    last_meal_each_friend_can_serve_index  = 5 + 5 * len(availability) * 3
    for meal in meal_serving_indexes:
        for i in range(number_of_meal_options_per_person): 
            index = first_meal_each_friend_can_serve_index + i
            while index < (last_meal_each_friend_can_serve_index +1 ):
                if meal == index:
                    day_index.append(i//2)
                index += number_of_meal_options_per_person

    # Allocating the friends to associated slot
    for i in range(len(meal_serving_indexes)):
        if breakfast_or_dinner[i] == 'b':
            breakfast_allocation[day_index[i]] = individuals[i]
        elif breakfast_or_dinner[i] == 'd':
            dinner_allocation[day_index[i]] = individuals[i]
    
    # Allocating the restaurant orders
    for i in range(len(breakfast_allocation)):
        if breakfast_allocation[i] == None:
            breakfast_allocation[i] = 5
    for i in range(len(dinner_allocation)):
        if dinner_allocation[i] == None:
            dinner_allocation[i] = 5
    
    # Output values
    final_output = (breakfast_allocation, dinner_allocation)
    return final_output

"""
=====|Task 1 End|====================================================================================
"""



"""
This marks the start of Task 2 of Assignment 2
=====|Task 2 Start|==================================================================================
"""
def fixing_in_built_rounding(num):
    """
    Function to do proper rounding 

    Time complexity         : O(1)
    """
    whole_number = num // 1
    decimal = num % 1
    if decimal >= 0.5:
        return whole_number + 1
    else:
        return whole_number

"""
This is a special modified node data structure for this specific implementation of Trie
"""
class TrieNode:
    def __init__(self, character, depth = 0) -> None:
        self.character = character
        self.character_list = [None]*28 # $,' ',a,b ... y,z 
        self.prev = None 
        self.depth = depth
        self.visited = [False, False]

    def visited_Node_string_i(self, stringx):
        self.visited[ (stringx -1) ] = True

"""
This is a modified Trie data structure 
"""
class Trie:
    """
    This is the constructor for the modified Trie data structure

    Time complexity         : O((M+N)^2)
    Aux Space complexity    : O(N+M)
    """
    def __init__(self, string1, string2) -> None:
        self.root = TrieNode(';')
        self.string1 = string1 
        self.string2 = string2
        self.deepest_shared_node = None

        self.insert(string1, 1)
        self.insert(string2, 2)

    """
    Method to insert a string (and its suffixes) into the Trie
    """
    def insert(self, string_to_be_inserted, stringx):
        temp_string = ""
        index = 1

        # Insert all the values in backwards to simulate a proper suffix tree
        # example:
        # apple
        # pple
        # ple
        # ...
        while index <= len(string_to_be_inserted):
            try:
                self.insert_aux(temp_string, stringx)
                temp_string += string_to_be_inserted[index* (-1)]
                index += 1
            except:
                break
        self.insert_aux(temp_string, stringx)
    
    """
    This method is a method to help insert a given suffix string into the Trie
    """
    def insert_aux(self, string, stringx):
        current_node = self.root
        # Declare the root node has been visited
        current_node.visited_Node_string_i(stringx)

        # Declare a depth counter
        current_depth = 1
        # Declare a somewhat invalid index that should be replaced
        index = -1
        # Finding the index of the character in question
        for letter in string:
            if letter == '$':
                index = 0
            elif letter == ' ':
                index = 1
            else:
                index = ord(letter) - 97 + 2 # (ord('a') == 97) and +2 [$, ' ', a, ......]

            # Get the child node
            child_node = current_node.character_list[index]
            if child_node == None:
                current_node.character_list[index] = TrieNode(letter,current_depth)
                child_node = current_node.character_list[index]
                
                # If both the character from string1 and string2 is visited 
                if (current_node.visited[0] and current_node.visited[1]):
                    if self.deepest_shared_node is not None:
                        # Replace deepest shared node if there is even deeper node
                        if self.deepest_shared_node.depth <= current_node.depth:
                            self.deepest_shared_node = current_node
                        else:
                            self.deepest_shared_node = current_node
                # Update for backtracking
                child_node.prev = current_node
            
            # Increment depth
            current_depth += 1
            # Set visited
            child_node.visited_Node_string_i(stringx)
            # Update current node 
            current_node = child_node
        
        # Declare the index of terminal sign (0) hard coded sorry
        terminal_icon_index = 0
        # At end of string append a node with terminal 
        if current_node.character_list[terminal_icon_index] == None:
            current_node.character_list[terminal_icon_index] = TrieNode('$', current_depth)

        # Update for backtracking
        current_node.character_list[terminal_icon_index].prev = current_node
        
        # Replace deepest node if needed
        if (current_node.visited[0] and current_node.visited[1]):
            if self.deepest_shared_node is not None:
                if self.deepest_shared_node.depth <= current_node.depth:
                    self.deepest_shared_node = current_node
            else:
                self.deepest_shared_node = current_node

        # Finally append the terminal node
        current_node = current_node.character_list[terminal_icon_index]
        current_node.visited_Node_string_i(stringx)


"""
    This method is a method to compare the longest similarity between 2 strings

    Time complexity         : O((M+N)^2)
    Aux Space complexity    : O(N+M)
    """
def compare_subs(string_1, string_2):
    
    # Creating the Trie
    my_trie = Trie(string_1, string_2)

    
    output_string = ""
    # Terminate early and give appropriate output when where is no shared nodes
    if my_trie.deepest_shared_node == None:
        return ["", 0 , 0]
    else:
        # Find the deepest shared node in Trie
        temp_node = my_trie.deepest_shared_node
        temp_output_string = ""
        # Iterate until full print is bult
        while temp_node is not None:
            temp_output_string += temp_node.character
            temp_node = temp_node.prev
        # revere the string
        output_string = temp_output_string[:-1]
    
    # find the similarity percentages 
    percentage_string1 = int(fixing_in_built_rounding( len(output_string) / len(string_1) * 100 ) )
    percentage_string2 = int(fixing_in_built_rounding( len(output_string) / len(string_2) * 100 ) )
    
    # declare output
    output_list = [output_string, percentage_string1, percentage_string2]

    # return output
    return output_list

"""
=====|Task 2 End|====================================================================================
"""

    


        
                 
        










