import networkx as nx
import matplotlib.pyplot as plt
import random
from pydub import AudioSegment
from pydub.generators import Sine
from collections import deque

# Define musical notes for the C major scale
cmajor_scale = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

# Create the object TreeNode which consists of nodes in a binary tree
class TreeNode:
    # Initialize the attributes of TreeNode
    def __init__(self, value):
        # self = node object
        # value = value of a node
        self.value = value
        # Ensure that the tree can have at most two branches (left and right)
        # Set left and right to none, so a node will default to having no 'children' nodes
        self.left = None
        self.right = None

# Function to create a binary tree
def create_binary_tree_graph():
    # Create a binary tree with notes as values
    # Create the root of the binary tree
    root = TreeNode('C')
    # Create left and right children
    root.left = TreeNode('D')
    root.right = TreeNode('E')
    root.left.left = TreeNode('F')
    root.left.right = TreeNode('G')
    root.right.right = TreeNode('A')
    root.right.left = TreeNode('B')
    return root

# Function to create a cyclic graph
def create_cyclic_graph():
    # Shuffle the notes to create a random order 
    # (which will lead to a random melody)
    random.shuffle(cmajor_scale)
    # Create a directed graph
    G = nx.DiGraph()
    # Add verticies/nodes to the graph, each vertex representing one musical note from the C major scale
    G.add_nodes_from(cmajor_scale)
    # Loop through the musical notes and edd edges between every pair of verticies
    # Edges are meant to represent the transitions between notes
    for i in range(len(cmajor_scale) - 1):
        G.add_edge(cmajor_scale[i], cmajor_scale[i + 1])
    # Add an edge to represent the transition from the last note back to the first
    # (To ensure that the graph makes a full cycle)
    G.add_edge(cmajor_scale[-1], cmajor_scale[0])
    return G

# Function to draw out a binary tree graph
def draw_binary_tree_graph(tree):
    # Create an empty, undirected graph
    G = nx.Graph()
    # Add edges from a TreeNode binary tree to G, using the traverse_tree function
    traverse_tree(tree, G)
    # Create a static layout of all the nodes (musical notes)
    pos = {
        'C': (0, 0),
        'D': (-1, -1),
        'E': (1, -1),
        'F': (-2, -2),
        'G': (-0.5, -2),
        'A': (2, -2),
        'B': (0.5, -2),
    }
    # Add labels to the edges of the root node to indicate if they are branching out left or right
    edge_labels = {(tree.value, tree.left.value): 'L', (tree.value, tree.right.value): 'R'}
    
    # 'Draw' the graph using Matplotlib
    # Input labels, node size, node color, font size, font color, and font weight
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_color="black",
            font_weight="bold", arrowsize=20)
    # 'Draw' the labels on the edges of the graph using edge_labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # Show the graph using Matplotlib
    plt.show()

# Function to traverse a binary tree graph and add edges
def traverse_tree(node, graph):
    # Check if the node has a left child
    if node.left:
        # Add an edge between the current node and left child
        graph.add_edge(node.value, node.left.value)
        # Recursively call traverse_tree to traverse the next left node
        traverse_tree(node.left, graph)
    # Check if the node has a right child
    if node.right:
        # Add an edge between the current node and right child
        graph.add_edge(node.value, node.right.value)
        # Recursively call traverse_tree to traverse the next right node
        traverse_tree(node.right, graph)

# Function to draw the graph image
def draw_cyclic_graph(G):
    # spring_layout is a layout algorithm in NetworkX
    # Calculate the layout for graph G with a spacing factor of 2 between nodes
    pos = nx.spring_layout(G, k=2)
    # Get the weight of edges from the graph input
    edge_labels = nx.get_edge_attributes(G, 'weight')
    # 'Draw' graph using Matplotlib
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_color="black",
            font_weight="bold", arrowsize=20)
    # Draw labels on the edges
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # Show the graph on the computer screen (Matplotlib function)
    plt.show()

# Function that generates a melody based on a binary tree traversal type
# Traversal types: preorder, postorder, inorder, bfs, dfs
def generate_binary_tree_melody(root, traversal_type):
    # Create an empty list that will hold musical notes
    melody = []

    # Nested function that recursively calls traverse_tree with a node
    # depending on the traversal_type input.
    def traverse_tree(node):
        nonlocal melody
        # Check if node != None 
        # (in that case we have reached a leaf of the binary tree)
        if node:
            # Check traversal_type input (preorder, inorder, postorder)
            # Append the node to the melody list based on traversal_type
            if traversal_type == 'preorder':
                # Appends the current node's value to the melody list.
                melody.append(node.value)
            # Recursively calls traverse_tree on the left child of the current node.
            traverse_tree(node.left)
            if traversal_type == 'inorder':
                melody.append(node.value)
            # Recursively calls traverse_tree on the right child of the current node.
            traverse_tree(node.right)
            if traversal_type == 'postorder':
                melody.append(node.value)

    # Nested function that performs a breadth first search traversal
    def bfs_traversal(node):
        nonlocal melody
        # Initialize a double-ended queue starting with the node input
        queue = deque([node])

        # Loop until the queue is empty
        while queue:
            # Remove and return leftmost node in the deque
            # Simulates visiting nodes level by level
            current_node = queue.popleft()
            # Append value of the dequeued node to the melody list
            melody.append(current_node.value)
            # Create variable to store the left and right children of the current node
            neighbors = [current_node.left, current_node.right]
            # Remove any 'None' values from neighbors
            neighbors = [n for n in neighbors if n is not None]
            # 'Extend queue' (continute BFS traversal)
            queue.extend(neighbors)

    # Nested function that performs a depth first search traversal
    def dfs_traversal(node):
        nonlocal melody
        # Check if node != None
        if node:
            # Append value of the current node to the melody list
            melody.append(node.value)
            # Recursively call dfs_traversal to the left and right children of the current node
            dfs_traversal(node.left)
            dfs_traversal(node.right)

    # Check the traversal_type (bfs or dfs)
    # Call the specified traversal function on the root node
    if traversal_type == 'bfs':
        bfs_traversal(root)
    elif traversal_type == 'dfs':
        dfs_traversal(root)
    else:
        traverse_tree(root)

    # Return the list of nodes (musical notes)
    return melody

# Function that generates a melody based on traversing through a cyclic graph
def generate_cyclic_melody(graph, start_note, length):
    # Create an empty list that will store the nodes 
    melody = []
    # Create a list with the start_note 
    # (to track the nodes that need to be traversed)
    stack = [start_note]

    # Loop until stack is empty or desired length is reached
    while stack and len(melody) < length:
        # Remove the current node from stack
        current_note = stack.pop()
        # Append the current note to the melody list
        melody.append(current_note)
        # Gets the neighbors of the current node in the graph
        neighbors = list(graph.neighbors(current_note))
        
        if neighbors:
            # Make the next_note the first neighbor from the list
            next_note = neighbors[0] 
            # Appends next_note to stack to be explored
            stack.append(next_note)
    # Return the list of musical notes
    return melody


def note_to_freq(note):
    # Assign the musical notes to frequencies (Hz)
    note_freq = {'C': 261.63, 'D': 293.66, 'E': 329.63, 'F': 349.23, 'G': 392.00, 'A': 440.00, 'B': 493.88}

    # Check if the input note is in note_freq
    if note in note_freq:
        # Return the frequency of the input note
        return note_freq[note]
    else:
        # Raise a ValueError if the input note is not in the c-major scale
        raise ValueError(f"Note {note} not found in the C-Major scale.")

# Define duration for one note (miliseconds)
duration = 1000  

# Function to generate an audio segment using a generated melody
# (Note: do not add any spaces to 'graph_type' input)
def melody_to_audio(graph_type, melody):
    # Print the generated melody into the console.
    print("Generated", graph_type + " Melody:",melody)
    # Create an audio segment
    audio = AudioSegment.silent(duration=duration)
    # Iterate over each note in melody
    for note in melody:
        # Get the frequency for the current note
        frequency = note_to_freq(note)
        # Add a sine wave audio segment to audio
        audio += Sine(frequency).to_audio_segment(duration=duration, volume=-8)
    # Export the audio segment to a WAV file
    # Replace line below with desired export location:
    # audio.export("C:\\Users\\abby-\\OneDrive\\Documents\\music-graphs\\audio\\" + graph_type + ".wav", format="wav")

# Create a binary tree graph
binaryTree = create_binary_tree_graph()

# Generate a melody using preorder traversal
melody_to_audio("Preorder", generate_binary_tree_melody(binaryTree, 'preorder'))

# Generate a melody using postorder traversal
melody_to_audio("Postorder", generate_binary_tree_melody(binaryTree, 'postorder'))

# Generate a melody using inorder traversal
melody_to_audio("Inorder", generate_binary_tree_melody(binaryTree, 'inorder'))

# Generate a melody using breadth-first search
melody_to_audio("BFS", generate_binary_tree_melody(binaryTree, 'bfs'))

# Generate a melody using depth-first search
melody_to_audio("DFS", generate_binary_tree_melody(binaryTree, 'dfs'))

# Create an image of the binary tree graph
draw_binary_tree_graph(binaryTree)

# Generate a cyclic, hamiltonian graph
G = create_cyclic_graph()

# Generate a melody starting from the note 'C' with a length of 8 (full cycle of graph)
melody_to_audio("Hamiltonian", generate_cyclic_melody(G, 'C', 8))

# Generate a melody starting from the note 'C' with a length of 4 (a path in the cyclic graph)
melody_to_audio("Path", generate_cyclic_melody(G, 'C', 4))

# Create an image of the cyclic graph
draw_cyclic_graph(G)