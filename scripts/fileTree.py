import os

# Generate the directory tree structure
tree_output = '\n'.join(f for f in os.popen('tree /f').readlines())

# Get the directory of the script file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the output file path in the script's directory
output_file_path = os.path.join(script_dir, 'tree_output.txt')

# Save the output to the file
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(tree_output)

print(f"Tree output saved to {output_file_path}")
