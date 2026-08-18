import ast
import os

def check_docstring(node):
    docstring = ast.get_docstring(node)
    if not docstring:
        return False
    # Simple heuristic for Google style: might contain "Args:" or "Returns:" if applicable
    # Just checking if there is a docstring for now.
    return True

def analyze_directory(directory):
    missing_docs = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read())
                    except SyntaxError:
                        continue
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if not check_docstring(node):
                                missing_docs.append((path, node.name, node.lineno))
    return missing_docs

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Find functions missing docstrings.")
    parser.add_argument("dir", help="Directory to scan for Python files with missing docstrings.")
    args = parser.parse_args()
    src_dir = args.dir
    missing = analyze_directory(src_dir)
    print(f"Total missing: {len(missing)}")
    for m in missing:
        print(f"{m[0]}: {m[1]} at line {m[2]}")
