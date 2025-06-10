#!/usr/bin/env python3
"""
HRRR Project Context Documentation Generator
Generates a comprehensive context document of the project structure and code.
"""

import os
import json
from pathlib import Path
import fnmatch
from typing import List, Set
import re


class ProjectContextGenerator:
    """Generate comprehensive project context documentation"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.gitignore_patterns = self._load_gitignore()
        
    def _load_gitignore(self) -> List[str]:
        """Load and parse .gitignore patterns"""
        gitignore_path = self.project_root / ".gitignore"
        patterns = []
        
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#'):
                        # Handle negation patterns (!)
                        if line.startswith('!'):
                            continue  # We'll handle inclusions separately
                        patterns.append(line)
        
        return patterns
    
    def _is_ignored(self, path: Path) -> bool:
        """Check if a path should be ignored based on .gitignore patterns"""
        relative_path = path.relative_to(self.project_root)
        
        # Always ignore .git directory
        if '.git' in relative_path.parts:
            return True
            
        # Check against gitignore patterns
        for pattern in self.gitignore_patterns:
            # Convert gitignore pattern to fnmatch pattern
            if pattern.endswith('/'):
                # Directory pattern
                if relative_path.is_dir() and fnmatch.fnmatch(str(relative_path), pattern[:-1]):
                    return True
                # Also check if any parent directory matches
                for parent in relative_path.parents:
                    if fnmatch.fnmatch(str(parent), pattern[:-1]):
                        return True
            else:
                # File or directory pattern
                if fnmatch.fnmatch(str(relative_path), pattern):
                    return True
                if fnmatch.fnmatch(relative_path.name, pattern):
                    return True
                    
        return False
    
    def _should_include_file_content(self, path: Path) -> bool:
        """Check if file content should be included in documentation"""
        if not path.is_file():
            return False
            
        # Include Python files
        if path.suffix == '.py':
            return True
            
        # Include JSON configuration files
        if path.suffix == '.json':
            return True
            
        # Include markdown files
        if path.suffix in ['.md', '.txt']:
            return True
            
        # Include YAML files
        if path.suffix in ['.yml', '.yaml']:
            return True
            
        # Include requirements files
        if path.name in ['requirements.txt', 'environment.yml', 'pyproject.toml', 'setup.py']:
            return True
            
        return False
    
    def generate_file_tree(self, max_depth: int = None) -> str:
        """Generate a file tree representation respecting .gitignore"""
        tree_lines = []
        
        def _add_tree_line(path: Path, prefix: str = "", is_last: bool = True, depth: int = 0):
            if max_depth is not None and depth > max_depth:
                return
                
            if self._is_ignored(path):
                return
                
            # Determine the tree symbols
            if depth == 0:
                symbol = ""
                new_prefix = ""
            else:
                symbol = "└── " if is_last else "├── "
                new_prefix = prefix + ("    " if is_last else "│   ")
            
            # Add current item
            if path.is_dir():
                tree_lines.append(f"{prefix}{symbol}{path.name}/")
                
                # Get children and sort them (directories first, then files)
                try:
                    children = list(path.iterdir())
                    children.sort(key=lambda x: (x.is_file(), x.name.lower()))
                    
                    # Filter out ignored children
                    children = [child for child in children if not self._is_ignored(child)]
                    
                    # Add children
                    for i, child in enumerate(children):
                        is_last_child = (i == len(children) - 1)
                        _add_tree_line(child, new_prefix, is_last_child, depth + 1)
                        
                except PermissionError:
                    tree_lines.append(f"{new_prefix}[Permission Denied]")
            else:
                tree_lines.append(f"{prefix}{symbol}{path.name}")
        
        tree_lines.append(f"{self.project_root.name}/")
        
        # Start with project root contents
        try:
            children = list(self.project_root.iterdir())
            children.sort(key=lambda x: (x.is_file(), x.name.lower()))
            children = [child for child in children if not self._is_ignored(child)]
            
            for i, child in enumerate(children):
                is_last_child = (i == len(children) - 1)
                _add_tree_line(child, "", is_last_child, 1)
                
        except PermissionError:
            tree_lines.append("[Permission Denied]")
            
        return "\n".join(tree_lines)
    
    def collect_files_for_content(self) -> List[Path]:
        """Collect all files that should have their content included"""
        files = []
        
        def _collect_files(path: Path):
            if self._is_ignored(path):
                return
                
            if path.is_file() and self._should_include_file_content(path):
                files.append(path)
            elif path.is_dir():
                try:
                    for child in path.iterdir():
                        _collect_files(child)
                except PermissionError:
                    pass
        
        _collect_files(self.project_root)
        
        # Sort files by category and then by path
        def sort_key(file_path: Path):
            relative = file_path.relative_to(self.project_root)
            
            # Category priority
            if file_path.suffix == '.py':
                if file_path.parent.name == 'derived_params':
                    return (1, str(relative))  # Derived params first
                elif file_path.name.endswith('_processor.py') or file_path.name.endswith('_refactored.py'):
                    return (2, str(relative))  # Main processors second
                else:
                    return (3, str(relative))  # Other Python files
            elif file_path.suffix == '.json':
                return (4, str(relative))  # JSON configs
            elif file_path.suffix in ['.md', '.txt']:
                return (5, str(relative))  # Documentation
            else:
                return (6, str(relative))  # Everything else
                
        files.sort(key=sort_key)
        return files
    
    def generate_file_content_section(self, file_path: Path) -> str:
        """Generate content section for a single file"""
        relative_path = file_path.relative_to(self.project_root)
        
        # Header
        header = f"{'='*80}\n"
        header += f"FILE: {relative_path}\n"
        header += f"{'='*80}\n"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Add line numbers for Python and JSON files
            if file_path.suffix in ['.py', '.json']:
                lines = content.split('\n')
                numbered_lines = [f"{i+1:4d}  {line}" for i, line in enumerate(lines)]
                content = '\n'.join(numbered_lines)
            
            return f"{header}\n{content}\n\n"
            
        except UnicodeDecodeError:
            return f"{header}\n[Binary file - content not displayed]\n\n"
        except Exception as e:
            return f"{header}\n[Error reading file: {e}]\n\n"
    
    def generate_full_context(self, output_file: str = "PROJECT_CONTEXT.md") -> str:
        """Generate complete project context documentation"""
        
        # Determine if this is a plain text file
        is_txt_file = output_file.lower().endswith('.txt')
        
        if is_txt_file:
            # Plain text format
            context = "HRRR SEVERE WEATHER PROCESSING PROJECT - COMPLETE CONTEXT\n"
            context += "=" * 70 + "\n\n"
            context += f"Generated from: {self.project_root}\n"
            context += f"Generated on: {os.popen('date').read().strip()}\n\n"
            
            # Table of Contents
            context += "TABLE OF CONTENTS\n"
            context += "-" * 20 + "\n\n"
            context += "1. Project Structure\n"
            context += "2. Source Code Files\n"
            context += "   - Main Processing Scripts\n"
            context += "   - Derived Parameters\n"
            context += "   - Configuration Files\n"
            context += "   - Documentation\n\n"
            
            # Project Structure
            context += "PROJECT STRUCTURE\n"
            context += "-" * 20 + "\n\n"
            context += self.generate_file_tree(max_depth=4)
            context += "\n\n"
            
            # File Contents
            context += "SOURCE CODE FILES\n"
            context += "-" * 20 + "\n\n"
        else:
            # Markdown format
            context = "# HRRR Severe Weather Processing Project - Complete Context\\n\\n"
            context += f"Generated from: {self.project_root}\\n"
            context += f"Generated on: {os.popen('date').read().strip()}\\n\\n"
            
            # Table of Contents
            context += "## Table of Contents\\n\\n"
            context += "1. [Project Structure](#project-structure)\\n"
            context += "2. [Source Code Files](#source-code-files)\\n"
            context += "   - [Main Processing Scripts](#main-processing-scripts)\\n"
            context += "   - [Derived Parameters](#derived-parameters)\\n"
            context += "   - [Configuration Files](#configuration-files)\\n"
            context += "   - [Documentation](#documentation)\\n\\n"
            
            # Project Structure
            context += "## Project Structure\\n\\n"
            context += "```\\n"
            context += self.generate_file_tree(max_depth=4)
            context += "\\n```\\n\\n"
            
            # File Contents
            context += "## Source Code Files\\n\\n"
        
        files = self.collect_files_for_content()
        
        # Group files by category for better organization
        main_scripts = []
        derived_params = []
        config_files = []
        docs = []
        other_files = []
        
        for file_path in files:
            relative = file_path.relative_to(self.project_root)
            
            if file_path.suffix == '.py':
                if file_path.parent.name == 'derived_params':
                    derived_params.append(file_path)
                elif ('processor' in file_path.name or 'smart' in file_path.name or 
                      file_path.name in ['field_registry.py', 'derived_parameters.py', 'field_templates.py']):
                    main_scripts.append(file_path)
                else:
                    other_files.append(file_path)
            elif file_path.suffix == '.json':
                config_files.append(file_path)
            elif file_path.suffix in ['.md', '.txt']:
                docs.append(file_path)
            else:
                other_files.append(file_path)
        
        # Add sections
        if main_scripts:
            if is_txt_file:
                context += "MAIN PROCESSING SCRIPTS\n" + "-" * 25 + "\n\n"
            else:
                context += "### Main Processing Scripts\\n\\n"
            for file_path in main_scripts:
                context += self.generate_file_content_section(file_path)
        
        if config_files:
            if is_txt_file:
                context += "CONFIGURATION FILES\n" + "-" * 20 + "\n\n"
            else:
                context += "### Configuration Files\\n\\n"
            for file_path in config_files:
                context += self.generate_file_content_section(file_path)
                
        if derived_params:
            if is_txt_file:
                context += "DERIVED PARAMETERS\n" + "-" * 18 + "\n\n"
            else:
                context += "### Derived Parameters\\n\\n"
            for file_path in derived_params:
                context += self.generate_file_content_section(file_path)
        
        if docs:
            if is_txt_file:
                context += "DOCUMENTATION\n" + "-" * 13 + "\n\n"
            else:
                context += "### Documentation\\n\\n"
            for file_path in docs:
                context += self.generate_file_content_section(file_path)
                
        if other_files:
            if is_txt_file:
                context += "OTHER FILES\n" + "-" * 11 + "\n\n"
            else:
                context += "### Other Files\\n\\n"
            for file_path in other_files:
                context += self.generate_file_content_section(file_path)
        
        # Save to file
        output_path = self.project_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(context)
            
        print(f"✅ Project context documentation generated: {output_path}")
        print(f"📊 Total files included: {len(files)}")
        print(f"   - Main scripts: {len(main_scripts)}")
        print(f"   - Derived params: {len(derived_params)}")
        print(f"   - Config files: {len(config_files)}")
        print(f"   - Documentation: {len(docs)}")
        print(f"   - Other files: {len(other_files)}")
        
        return str(output_path)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate project context documentation")
    parser.add_argument("--output", "-o", default="PROJECT_CONTEXT.md",
                       help="Output file name (default: PROJECT_CONTEXT.md)")
    parser.add_argument("--tree-depth", type=int, default=4,
                       help="Maximum depth for file tree (default: 4)")
    
    args = parser.parse_args()
    
    generator = ProjectContextGenerator()
    output_path = generator.generate_full_context(args.output)
    
    print(f"\\n🎉 Context documentation ready: {output_path}")


if __name__ == "__main__":
    main()