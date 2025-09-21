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

        # Patterns for function detection
        self.function_pattern = re.compile(
            r'^(?!.*(?:inline|template|static.*inline))'  # Not inline/template
            r'(?:\s*(?:virtual|static|explicit|constexpr)?)*\s*'  # Optional modifiers
            r'(?:(?:const\s+)?(?:unsigned\s+)?(?:signed\s+)?'     # Optional type modifiers
            r'(?:void|bool|int|float|double|char|wchar_t|size_t|'  # Basic types
            r'std::\w+|[A-Z]\w*(?:::\w+)*)\s*[\*&]*)\s+'          # Complex types
            r'([A-Za-z_]\w*(?:::\w+)*)'                           # Function name
            r'\s*\([^)]*\)'                                       # Parameters
            r'(?:\s*const)?'                                      # Optional const
            r'\s*\{'                                              # Opening brace
            r'(?!\s*(?://|/\*))',                                 # Not followed by comment
            re.MULTILINE
        )

        # Class method pattern
        self.method_pattern = re.compile(
            r'^(?:\s*(?:virtual|static|explicit|constexpr|inline)?)*\s*'
            r'(?:(?:const\s+)?(?:unsigned\s+)?(?:signed\s+)?'
            r'(?:void|bool|int|float|double|char|wchar_t|size_t|'
            r'std::\w+|[A-Z]\w*(?:::\w+)*)\s*[\*&]*)\s+'
            r'([A-Z]\w*)::'                                       # Class name
            r'([A-Za-z_]\w*)'                                     # Method name
            r'\s*\([^)]*\)'
            r'(?:\s*const)?'
            r'\s*\{'
            r'(?!\s*(?://|/\*))',
            re.MULTILINE
        )

        # Patterns to skip
        self.skip_patterns = [
            re.compile(r'^\s*[A-Za-z_]\w*\s*\(\s*\)\s*\{\s*\}'),     # Empty functions
            re.compile(r'^\s*[A-Za-z_]\w*\s*\([^)]*\)\s*:\s*'),      # Constructor initializer
            re.compile(r'^\s*~[A-Za-z_]\w*\s*\('),                   # Destructors
            re.compile(r'^\s*operator[^\(]*\('),                     # Operators
            re.compile(r'^\s*(?:get|set)[A-Z]\w*\s*\([^)]*\)\s*\{.*\}\s*$'),  # Simple getters/setters
        ]

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

    def should_skip_function(self, func_text: str) -> bool:
        """Check if function should be skipped from instrumentation."""
        for pattern in self.skip_patterns:
            if pattern.search(func_text):
                return True
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
        """Add Tracy include to file if not present."""
        if self.tracy_include in content:
            return content

        # Find the first non-comment, non-blank line after existing includes
        lines = content.split('\n')
        insert_pos = 0

        for i, line in enumerate(lines):
            if line.strip().startswith('#include'):
                insert_pos = i + 1
            elif line.strip() and not line.strip().startswith('//') and not line.strip().startswith('/*'):
                break

        lines.insert(insert_pos, self.tracy_include)
        return '\n'.join(lines)

    def instrument_function(self, match: re.Match, file_path: Path) -> str:
        """Add Tracy instrumentation to a single function."""
        full_match = match.group(0)

        # Skip if already instrumented or should be skipped
        if 'PROFILE_' in full_match or self.should_skip_function(full_match):
            return full_match

        # Determine if it's a method or function
        if '::' in full_match:
            # Method
            method_match = self.method_pattern.search(full_match)
            if method_match:
                class_name = method_match.group(1)
                func_name = method_match.group(2)
                zone_name = self.get_zone_name(file_path, func_name, class_name)
            else:
                zone_name = 'PROFILE_ZONE()'
        else:
            # Regular function
            func_match = self.function_pattern.search(full_match)
            if func_match:
                func_name = func_match.group(1)
                zone_name = self.get_zone_name(file_path, func_name)
            else:
                zone_name = 'PROFILE_ZONE()'

        # Find the opening brace and add instrumentation
        brace_pos = full_match.find('{')
        if brace_pos == -1:
            return full_match

        before_brace = full_match[:brace_pos + 1]
        after_brace = full_match[brace_pos + 1:]

        # Add the profiling zone right after the opening brace
        instrumented = f"{before_brace}\n    {zone_name};\n{after_brace}"

        return instrumented

    def instrument_file(self, file_path: Path, preview_mode: bool = False) -> Tuple[bool, str]:
        """Instrument a single file with Tracy profiling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return False, f"Failed to read {file_path}: {e}"

        if self.is_already_instrumented(content):
            return False, f"File {file_path} is already instrumented"

        original_content = content

        # Add Tracy include
        content = self.add_tracy_include(content)

        # Add instrumentation marker
        content = f"{self.instrumented_marker}\n{content}"

        # Instrument functions
        instrumented_count = 0

        # Function instrumentation
        def replace_func(match):
            nonlocal instrumented_count
            instrumented_count += 1
            return self.instrument_function(match, file_path)

        content = self.function_pattern.sub(replace_func, content)
        content = self.method_pattern.sub(replace_func, content)

        if instrumented_count == 0:
            return False, f"No functions found to instrument in {file_path}"

        if preview_mode:
            return True, f"Would instrument {instrumented_count} functions in {file_path}"

        # Write instrumented file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
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
            print(f"✓ {message}")
        else:
            print(f"✗ {message}")

    print(f"\nProcessed {success_count}/{len(files)} files successfully")

    if args.mode == 'preview':
        print("\nThis was a preview. Use --mode=auto to actually instrument files.")
        print("Use --backup to create backups before instrumentation.")

if __name__ == '__main__':
    main()