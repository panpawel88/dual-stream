#!/usr/bin/env python3
"""
Tracy Profiler Automatic Instrumentation Script

This script automatically adds Tracy profiling annotations to C++ source files.
It detects function definitions and adds appropriate ZoneScopedN macros for
comprehensive performance analysis.

Usage:
    python tracy_instrumentor.py --mode=auto --backup      # Automatically instrument all files
    python tracy_instrumentor.py --mode=preview           # Preview changes without modifying files
    python tracy_instrumentor.py --mode=clean             # Remove Tracy instrumentation
    python tracy_instrumentor.py --file=path/to/file.cpp  # Instrument specific file
"""

import os
import re
import sys
import argparse
import shutil
import glob
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple

class TracyInstrumentor:
    def __init__(self, src_dir: str = "src", backup_dir: str = "backup"):
        self.src_dir = Path(src_dir)
        self.backup_dir = Path(backup_dir)
        self.instrumented_marker = "// TRACY_INSTRUMENTED"
        self.tracy_include = '#include "core/TracyProfiler.h"'

        # Simple patterns for fast detection - no catastrophic backtracking
        self.function_start_pattern = re.compile(r'^[^/\n]*\{\s*$', re.MULTILINE)
        self.class_method_pattern = re.compile(r'([A-Z]\w*)::', re.MULTILINE)
        self.identifier_pattern = re.compile(r'\b([A-Za-z_]\w*)\s*\(')

        # Fast pre-filters
        self.comment_line = re.compile(r'^\s*(?://|/\*|\*)')
        self.preprocessor_line = re.compile(r'^\s*#')
        self.empty_line = re.compile(r'^\s*$')

        # Context tracking patterns
        self.extern_c_start = re.compile(r'extern\s+"C"\s*\{')
        self.extern_c_end = re.compile(r'\}')
        self.extern_c_inline = re.compile(r'extern\s+"C"\s*\{[^}]*\}')
        self.switch_statement = re.compile(r'^\s*switch\s*\(', re.MULTILINE)

        # Patterns to skip
        self.skip_patterns = [
            re.compile(r'^\s*[A-Za-z_]\w*\s*\(\s*\)\s*\{\s*\}'),     # Empty functions
            re.compile(r'^\s*[A-Za-z_]\w*\s*\([^)]*\)\s*:\s*'),      # Constructor initializer
            re.compile(r'^\s*~[A-Za-z_]\w*\s*\('),                   # Destructors
            re.compile(r'^\s*operator[^\(]*\('),                     # Operators
            re.compile(r'^\s*(?:get|set)[A-Z]\w*\s*\([^)]*\)\s*\{.*\}\s*$'),  # Simple getters/setters
            re.compile(r'^\s*case\s+[^:]+:'),                       # Switch cases
            re.compile(r'^\s*default\s*:'),                         # Default cases
            re.compile(r'^\s*\{[^}]*\}\s*,?\s*$'),                   # Array/struct initializers
            re.compile(r'=\s*\{'),                                  # Assignment with braces
            re.compile(r'^\s*\[\s*[^\]]*\]\s*\([^)]*\)\s*\{'),      # Lambda functions
            re.compile(r'^\s*auto\s+[^=]*=\s*\[[^\]]*\]'),          # Lambda assignments
        ]

        # Local type definition patterns
        self.local_struct_pattern = re.compile(r'^\s*struct\s+[A-Za-z_]\w*\s*\{')
        self.local_class_pattern = re.compile(r'^\s*class\s+[A-Za-z_]\w*\s*\{')
        self.local_enum_pattern = re.compile(r'^\s*enum\s+(?:class\s+)?[A-Za-z_]\w*\s*\{')
        self.local_union_pattern = re.compile(r'^\s*union\s+[A-Za-z_]\w*\s*\{')

        # String literal patterns
        self.raw_string_start = re.compile(r'R"([^(]*)\(')
        self.string_literal = re.compile(r'"([^"\\]|\\.)*"')

        # Files to exclude from instrumentation
        self.exclude_files = {
            'pch.cpp', 'pch.h',
            'TracyProfiler.cpp', 'TracyProfiler.h'
        }

        # Subsystem-specific profiling zones
        self.subsystem_zones = {
            'video/decode': 'PROFILE_VIDEO_DECODE',
            'video/demux': 'PROFILE_VIDEO_DEMUX',
            'video/switching': 'PROFILE_VIDEO_SWITCH',
            'rendering': 'PROFILE_RENDER',
            'camera': 'PROFILE_CAMERA_CAPTURE',
            'ui': 'PROFILE_UI_DRAW'
        }

    def should_skip_function_info(self, func_info: Dict[str, str]) -> bool:
        """Check if function should be skipped from instrumentation based on parsed info."""
        func_name = func_info['function_name']
        signature = func_info['signature']

        # Skip destructors
        if func_name.startswith('~'):
            return True

        # Skip operators
        if 'operator' in signature:
            return True

        # Skip constructors with initializer lists
        if ':' in signature and func_info['class_name'] and func_name == func_info['class_name']:
            return True

        # Skip simple getters/setters (single line)
        if (func_name.startswith('get') or func_name.startswith('set')) and signature.count('\n') <= 1:
            return True

        # Skip if already instrumented
        if 'PROFILE_' in signature:
            return True

        return False

    def is_in_context(self, lines: List[str], line_idx: int, context_type: str) -> bool:
        """Check if current line is within a specific context (extern C or switch)."""
        if context_type == 'extern_c':
            # Look backward for extern "C" {
            extern_c_depth = 0
            for i in range(line_idx, -1, -1):
                line = lines[i].strip()
                if self.extern_c_end.match(line):
                    extern_c_depth += 1
                elif self.extern_c_start.match(line):
                    extern_c_depth -= 1
                    if extern_c_depth < 0:
                        return True
            return False
        elif context_type == 'switch':
            # Look backward for switch statement
            brace_depth = 0
            for i in range(line_idx, -1, -1):
                line = lines[i].strip()
                if line.endswith('}'):
                    brace_depth += 1
                elif line.endswith('{'):
                    brace_depth -= 1
                    if brace_depth < 0 and self.switch_statement.search(line):
                        return True
                elif brace_depth == 0 and self.switch_statement.search(line):
                    return True
            return False
        return False

    def is_valid_function_signature(self, func_info: Dict[str, str], lines: List[str], line_idx: int) -> bool:
        """Enhanced validation of function signatures."""
        signature = func_info['signature']
        func_name = func_info['function_name']

        # Basic validation
        if not func_name or len(func_name) < 2:
            return False

        # Check for patterns that indicate this is not a function
        for pattern in self.skip_patterns:
            if pattern.search(signature):
                return False

        # Must have parentheses
        if '(' not in signature or ')' not in signature:
            return False

        # Check context - skip if in extern "C" or switch
        if self.is_in_context(lines, line_idx, 'extern_c'):
            return False
        if self.is_in_context(lines, line_idx, 'switch'):
            return False

        # Skip nested functions (local classes, lambdas in functions)
        if self.is_nested_function(lines, line_idx):
            return False

        # Skip functions that are too close to previous instrumentation
        if self.has_recent_tracy_annotation(lines, line_idx):
            return False

        # Skip if we're inside a local type definition (struct, class, enum, union)
        if self.is_inside_local_type_definition(lines, line_idx):
            return False

        # Skip if we're inside a string literal (especially raw string literals with shader code)
        if self.is_inside_string_literal(lines, line_idx):
            return False

        # Ensure this looks like a real function declaration/definition
        # Must have return type or be a constructor/method
        has_return_type = any(keyword in signature for keyword in
                            ['void', 'int', 'bool', 'char', 'float', 'double', 'auto', 'static', 'virtual', 'inline'])
        is_constructor = func_info['class_name'] and func_name == func_info['class_name']
        is_method = func_info['class_name'] is not None

        return has_return_type or is_constructor or is_method

    def is_nested_function(self, lines: List[str], line_idx: int) -> bool:
        """Check if this function is nested inside another function."""
        brace_depth = 0

        # Look backward to see if we're already inside a function
        for i in range(line_idx - 1, max(0, line_idx - 50), -1):
            line = lines[i].strip()

            # Count braces
            brace_depth += line.count('}') - line.count('{')

            # If we're inside braces and find another opening brace with function-like signature
            if brace_depth < 0:
                if ('{' in line and '(' in line and ')' in line and
                    not line.startswith('//') and not line.startswith('/*')):
                    return True

        return False

    def has_recent_tracy_annotation(self, lines: List[str], line_idx: int) -> bool:
        """Check if there's already a Tracy annotation very close to this location."""
        # Look at nearby lines for existing Tracy annotations
        for i in range(max(0, line_idx - 3), min(len(lines), line_idx + 3)):
            if 'PROFILE_' in lines[i]:
                return True
        return False

    def is_inside_local_type_definition(self, lines: List[str], line_idx: int) -> bool:
        """Check if current line is inside a local type definition (struct, class, enum, union)."""
        # Look backward to see if we're inside a local type definition
        brace_depth = 0

        for i in range(line_idx - 1, max(0, line_idx - 20), -1):
            line = lines[i].strip()

            # Count braces to understand nesting
            brace_depth += line.count('}') - line.count('{')

            # If we find an opening brace with local type definition pattern
            if brace_depth < 0:
                # Check if this line contains a local type definition
                if (self.local_struct_pattern.match(line) or
                    self.local_class_pattern.match(line) or
                    self.local_enum_pattern.match(line) or
                    self.local_union_pattern.match(line)):
                    return True

                # If we hit another opening brace that's not a type definition, we're not in one
                if '{' in line:
                    break

        return False

    def is_inside_string_literal(self, lines: List[str], line_idx: int) -> bool:
        """Check if current line is inside a string literal (especially raw string literals)."""
        # Look backward to see if we're inside a raw string literal
        for i in range(line_idx - 1, max(0, line_idx - 50), -1):
            line = lines[i]

            # Check for raw string literal start
            raw_match = self.raw_string_start.search(line)
            if raw_match:
                delimiter = raw_match.group(1)
                end_pattern = f'){delimiter}"'

                # Check if we've found the closing delimiter
                found_end = False
                for j in range(i, line_idx + 1):
                    if end_pattern in lines[j]:
                        found_end = True
                        break

                # If we haven't found the end, we're inside the raw string
                if not found_end:
                    return True

            # For regular string literals, we need more sophisticated parsing
            # but since most issues are with raw strings, this should handle the main case

        return False

    def get_zone_name(self, file_path: Path, func_name: str, class_name: str = None) -> str:
        """Generate appropriate zone name based on file path and function."""
        # Check for subsystem-specific zones
        for subsystem, zone_macro in self.subsystem_zones.items():
            if subsystem in str(file_path):
                if class_name:
                    return f'PROFILE_ZONE_N("{class_name}::{func_name}")'
                else:
                    return f'PROFILE_ZONE_N("{func_name}")'

        # Default zone naming
        if class_name:
            return f'PROFILE_ZONE_N("{class_name}::{func_name}")'
        else:
            return f'PROFILE_ZONE_N("{func_name}")'

    def find_cpp_files(self, specific_file: Optional[str] = None) -> List[Path]:
        """Find all C++ source files to instrument."""
        if specific_file:
            return [Path(specific_file)]

        cpp_files = []
        for pattern in ['**/*.cpp', '**/*.cxx', '**/*.cc']:
            cpp_files.extend(self.src_dir.glob(pattern))

        # Filter out excluded files
        return [f for f in cpp_files if f.name not in self.exclude_files]

    def is_already_instrumented(self, content: str) -> bool:
        """Check if file is already instrumented."""
        return self.instrumented_marker in content or "PROFILE_ZONE" in content

    def add_tracy_include(self, content: str) -> str:
        """Add Tracy include to file if not present, avoiding extern C blocks."""
        if self.tracy_include in content:
            return content

        # Handle inline extern "C" blocks first by splitting them
        content = self._split_inline_extern_c_blocks(content)

        lines = content.split('\n')
        insert_pos = 0
        extern_c_depth = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Track extern "C" blocks with proper brace counting
            if self.extern_c_start.search(stripped):
                extern_c_depth += stripped.count('{')
                continue
            elif stripped == '}' and extern_c_depth > 0:
                extern_c_depth -= 1
                # Insert after extern "C" block ends completely
                if extern_c_depth == 0:
                    insert_pos = i + 1
                continue

            # Only process includes outside extern "C" blocks
            if extern_c_depth == 0:
                if stripped.startswith('#include'):
                    insert_pos = i + 1
                elif stripped and not stripped.startswith('//') and not stripped.startswith('/*') and not self.preprocessor_line.match(stripped):
                    break

        lines.insert(insert_pos, self.tracy_include)
        return '\n'.join(lines)

    def _split_inline_extern_c_blocks(self, content: str) -> str:
        """Split inline extern C blocks to separate lines for proper processing."""
        # Find all inline extern "C" { ... } patterns and split them
        def split_extern_c(match):
            full_match = match.group(0)
            # Split: extern "C" { -> includes -> }
            parts = []

            # Find the opening brace
            start_pos = full_match.find('{')
            if start_pos == -1:
                return full_match

            # Add extern "C" {
            parts.append(full_match[:start_pos + 1])

            # Find the closing brace
            end_pos = full_match.rfind('}')
            if end_pos == -1:
                return full_match

            # Extract content between braces
            middle_content = full_match[start_pos + 1:end_pos]

            # Split includes and other content
            for part in middle_content.split('#include'):
                if part.strip():
                    if not part.startswith(' '):
                        parts.append('#include' + part)
                    else:
                        parts.append('#include' + part)

            # Add closing brace
            parts.append('}')

            return '\n'.join(parts)

        return self.extern_c_inline.sub(split_extern_c, content)

    def parse_function_signature(self, lines: List[str], start_idx: int) -> Optional[Dict[str, str]]:
        """Parse function signature from lines starting at start_idx."""
        # Look backwards to find the start of the function
        signature_lines = []
        i = start_idx

        # Go back to find the function signature
        while i >= 0 and len(signature_lines) < 10:  # Limit search
            line = lines[i].strip()
            if not line or self.comment_line.match(line) or self.preprocessor_line.match(line):
                i -= 1
                continue

            signature_lines.insert(0, line)

            # Check if we have a complete signature
            full_sig = ' '.join(signature_lines)
            if '(' in full_sig and ')' in full_sig and '{' in full_sig:
                break
            i -= 1

        if not signature_lines:
            return None

        full_signature = ' '.join(signature_lines)

        # Extract function name and class if present
        class_name = None
        func_name = None

        # Check for class method
        class_match = self.class_method_pattern.search(full_signature)
        if class_match:
            class_name = class_match.group(1)
            # Find function name after ::
            after_class = full_signature[class_match.end():]
            func_match = self.identifier_pattern.search(after_class)
            if func_match:
                func_name = func_match.group(1)
        else:
            # Regular function - find last identifier before (
            func_match = self.identifier_pattern.search(full_signature)
            if func_match:
                func_name = func_match.group(1)

        if not func_name:
            return None

        return {
            'function_name': func_name,
            'class_name': class_name,
            'signature': full_signature
        }

    def instrument_file(self, file_path: Path, preview_mode: bool = False) -> Tuple[bool, str]:
        """Instrument a single file with Tracy profiling using optimized line-by-line processing."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            return False, f"Failed to read {file_path}: {e}"

        content = ''.join(lines)
        if self.is_already_instrumented(content):
            return False, f"File {file_path} is already instrumented"

        # First, add Tracy include using sophisticated method
        content_with_include = self.add_tracy_include(content)
        lines = content_with_include.splitlines()

        # Process line by line for fast instrumentation
        instrumented_lines = []
        instrumented_count = 0

        # Add instrumentation marker
        instrumented_lines.append(f"{self.instrumented_marker}\n")

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Fast pre-filtering
            if (not stripped or
                self.comment_line.match(stripped) or
                self.preprocessor_line.match(stripped)):

                instrumented_lines.append(line)
                i += 1
                continue

            # Check for function opening brace
            if '{' in stripped and stripped.endswith('{'):
                # First check if this line itself is a local type definition
                if (self.local_struct_pattern.match(stripped) or
                    self.local_class_pattern.match(stripped) or
                    self.local_enum_pattern.match(stripped) or
                    self.local_union_pattern.match(stripped)):
                    # This is a local type definition - skip instrumentation
                    instrumented_lines.append(line)
                else:
                    # Parse function signature
                    func_info = self.parse_function_signature(lines, i)

                    if (func_info and
                        not self.should_skip_function_info(func_info) and
                        self.is_valid_function_signature(func_info, lines, i)):
                        # Add the line with opening brace
                        instrumented_lines.append(line)

                        # Add Tracy instrumentation
                        zone_name = self.get_zone_name(file_path, func_info['function_name'], func_info['class_name'])
                        instrumented_lines.append(f"    {zone_name};\n")
                        instrumented_count += 1
                    else:
                        instrumented_lines.append(line)
            else:
                instrumented_lines.append(line)

            i += 1

        if instrumented_count == 0:
            return False, f"No functions found to instrument in {file_path}"

        if preview_mode:
            return True, f"Would instrument {instrumented_count} functions in {file_path}"

        # Write instrumented file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Join with newlines to preserve file structure
                f.write('\n'.join(instrumented_lines))
            return True, f"Instrumented {instrumented_count} functions in {file_path}"
        except Exception as e:
            return False, f"Failed to write {file_path}: {e}"

    def clean_file(self, file_path: Path) -> Tuple[bool, str]:
        """Remove Tracy instrumentation from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return False, f"Failed to read {file_path}: {e}"

        if not self.is_already_instrumented(content):
            return False, f"File {file_path} is not instrumented"

        # Remove instrumentation marker
        content = content.replace(f"{self.instrumented_marker}\n", "")

        # Remove Tracy include
        content = content.replace(f"{self.tracy_include}\n", "")

        # Remove profiling zones
        content = re.sub(r'\s*PROFILE_[^;]*;\n', '', content)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, f"Cleaned instrumentation from {file_path}"
        except Exception as e:
            return False, f"Failed to write {file_path}: {e}"

    def backup_files(self, file_paths: List[Path]) -> bool:
        """Create backup of files before instrumentation."""
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True)

        for file_path in file_paths:
            try:
                backup_path = self.backup_dir / file_path.relative_to(self.src_dir)
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)
            except Exception as e:
                print(f"Failed to backup {file_path}: {e}")
                return False

        return True

    def restore_files(self) -> bool:
        """Restore files from backup."""
        if not self.backup_dir.exists():
            print("No backup directory found")
            return False

        try:
            for backup_file in self.backup_dir.glob('**/*'):
                if backup_file.is_file():
                    src_file = self.src_dir / backup_file.relative_to(self.backup_dir)
                    src_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_file, src_file)
            return True
        except Exception as e:
            print(f"Failed to restore files: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Tracy Profiler Automatic Instrumentation")
    parser.add_argument('--mode', choices=['auto', 'preview', 'clean', 'restore'],
                       default='preview', help='Operation mode')
    parser.add_argument('--file', help='Specific file to instrument')
    parser.add_argument('--src-dir', default='src', help='Source directory')
    parser.add_argument('--backup', action='store_true', help='Create backup before instrumentation')
    parser.add_argument('--force', action='store_true', help='Force operation even if already instrumented')

    args = parser.parse_args()

    instrumentor = TracyInstrumentor(args.src_dir)

    if args.mode == 'restore':
        if instrumentor.restore_files():
            print("Files restored from backup successfully")
        else:
            print("Failed to restore files")
        return

    files = instrumentor.find_cpp_files(args.file)
    if not files:
        print("No C++ files found to process")
        return

    print(f"Found {len(files)} C++ files to process")

    if args.backup and args.mode != 'preview':
        print("Creating backup...")
        if not instrumentor.backup_files(files):
            print("Failed to create backup, aborting")
            return

    success_count = 0
    for file_path in files:
        if args.mode == 'clean':
            success, message = instrumentor.clean_file(file_path)
        else:
            success, message = instrumentor.instrument_file(file_path, args.mode == 'preview')

        if success:
            success_count += 1
            print(f"[OK] {message}")
        else:
            print(f"[FAIL] {message}")

    print(f"\nProcessed {success_count}/{len(files)} files successfully")

    if args.mode == 'preview':
        print("\nThis was a preview. Use --mode=auto to actually instrument files.")
        print("Use --backup to create backups before instrumentation.")

if __name__ == '__main__':
    main()