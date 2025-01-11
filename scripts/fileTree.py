import os

def generate_filtered_tree():
    # Run the tree command to get the directory structure with all files
    tree_output = os.popen('tree /f').readlines()

    # Filter the output to exclude JSON files and include only directories and Python files
    filtered_output = []
    for line in tree_output:
        # Include directories (lines that don't have a file extension) or .py files
        if line.strip().endswith(".py") or not any(ext in line for ext in [".json", ".txt", ".csv"]):
            filtered_output.append(line)

    return '\n'.join(filtered_output)

# Get the directory of the script file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the output file path in the script's directory
output_file_path = os.path.join(script_dir, 'tree_output.txt')

# Generate the filtered tree structure
filtered_tree = generate_filtered_tree()

# Save the filtered tree output to the file
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(filtered_tree)

print(f"Filtered tree output saved to {output_file_path}")
