#!/usr/bin/env python3
"""
Test Runner for AI Trading Agent
Simple script to run all tests with various options
"""

import sys
import os
import subprocess
from pathlib import Path

def run_data_ingestion_tests():
    """Run data ingestion tests"""
    print("Running Data Ingestion Tests")
    print("=" * 40)
    
    test_file = Path("tests/test_data_ingestion.py")
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(test_file)], 
                              capture_output=True, text=True, cwd=".")
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"❌ Error running tests: {str(e)}")
        return False

def run_unittest_discovery():
    """Run unittest discovery for all tests"""
    print("Running All Tests (unittest discovery)")
    print("=" * 40)
    
    try:
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"], 
                              capture_output=True, text=True, cwd=".")
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"❌ Error running unittest discovery: {str(e)}")
        return False

def check_project_structure():
    """Check if project structure is correct"""
    print("Checking Project Structure")
    print("=" * 40)
    
    required_files = [
        "config.py",
        "main.py",
        "data/data_ingestion.py",
        "models/model_trainer.py",
        "tests/test_data_ingestion.py"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
            print(f"✓ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path}")
    
    print(f"\nSummary: {len(existing_files)} files found, {len(missing_files)} missing")
    
    if missing_files:
        print("\nMissing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
    
    return len(missing_files) == 0

def main():
    """Main test runner"""
    print("AI Trading Agent Test Runner")
    print("=" * 50)
    
    # Check project structure
    structure_ok = check_project_structure()
    print()
    
    if not structure_ok:
        print("❌ Project structure is incomplete. Please ensure all required files exist.")
        return 1
    
    # Get command line arguments
    test_type = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if test_type == "data":
        # Run only data ingestion tests
        success = run_data_ingestion_tests()
    elif test_type == "unittest":
        # Run unittest discovery
        success = run_unittest_discovery()
    else:
        # Run all tests
        print("Running all available tests...\n")
        
        # First run data ingestion tests
        data_success = run_data_ingestion_tests()
        print()
        
        # Then run unittest discovery
        unittest_success = run_unittest_discovery()
        
        success = data_success and unittest_success
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    
    if success:
        print("✅ All tests passed successfully!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())