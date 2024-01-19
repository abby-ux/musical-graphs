# musical-graphs
Graph Melodies Code Documentation
Abigail Reese
Discrete Structures School Project

Overview
The Graph Melodies code is designed to generate melodies using either a cyclic graph or binary tree graph based off of different traversal methods. The code utilizes the NetworkX library for graph creation and visualization, Matplotlib for graph rendering, and PyDub for audio generation.


Table of Contents
1. Introduction
2. Installation
3. Usage
   1. Binary Tree Melodies
   2. Cyclic Graph Melodies
4. Troubleshooting
5. Support


Introduction
Graph Melodies code can create either a cyclic or binary tree graph. The code can render these graphs as images, as well as save their melodies as WAV audio files. Binary tree melodies are created by traversing the graph in a preorder, postorder, inorder, breadth-first search, and/or depth-first search traversal. Cyclic graph melodies are created by traversing the cyclic tree in a hamiltonian path, or by traversing a partial path of the graph. The binary tree musical note sequence is the same every time, while the cyclic graph generates notes in a random order. 


Installation
Install required libraries:
        pip install networkx matplotlib pydub


Usage
1. Create a Binary Tree Graph
# Create a binary tree graph
binaryTree = create_binary_tree_graph()
2. Generate melodies
# Generate a melody using preorder traversal
melody_to_audio("Preorder", generate_binary_tree_melody(binaryTree, 'preorder'))
# Generate a melody using postorder traversal
melody_to_audio("Postorder", generate_binary_tree_melody(binaryTree, 'postorder'))
# Generate a melody using inorder traversal
melody_to_audio("Inorder", generate_binary_tree_melody(binaryTree, 'inorder'))
# Generate a melody using breadth-first search
melody_to_audio("BFS", generate_binary_tree_melody(binaryTree, 'bfs')
# Generate a melody using depth-first search
melody_to_audio("DFS", generate_binary_tree_melody(binaryTree, 'dfs'))
3. Visualize the Binary Tree Graph
# Create an image of the binary tree graph
draw_binary_tree_graph(binaryTree)
4. Create a Cyclic Graph
# Generate a cyclic, hamiltonian graph
G = create_cyclic_graph()
5. Generate melodies
# Generate a melody starting from the note 'C' with a length of 8 (full cycle of graph)
melody_to_audio("Hamiltonian", generate_cyclic_melody(G, 'C', 8))
# Generate a melody starting from the note 'C' with a length of 4 (a path in the cyclic graph)
melody_to_audio("Path", generate_cyclic_melody(G, 'C', 4))
6. Visualize the Binary Tree Graph
# Create an image of the cyclic graph
draw_cyclic_graph(G)


Troubleshooting
If you encounter any error messages, check the details of the message. Ensure that required libraries are installed, and that you input notes from the C-Major scale only.


Support
For further support, please contact reese.a@northeastern.edu