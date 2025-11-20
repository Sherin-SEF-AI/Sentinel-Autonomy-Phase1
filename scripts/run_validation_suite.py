#!/usr/bin/env python3
"""
Comprehensive validation suite for SENTINEL system.

Runs all integration, performance, reliability, and accuracy validation tests.
Generates a summary report of all validation results.

Usage:
    python scripts/run_validation_suite.py [--quick] [--report output.txt]
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path
import logging


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationSuite:
    """Comprehensive validation test suite"""
    
    def __init__(self, quick_mode=False, report_file=None):
        self.quick_mode = quick_mode
        self.report_file = report_file
        self.results = {}
        
    def run_test_suite(self, test_file, suite_name):
        """Run a test suite and capture results"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {suite_name}")
        logger.info(f"{'='*60}\n")
        
        start_time = time.time()
        
        try:
            # Run pytest
            cmd = [
                sys.executable, "-m", "pytest",
                test_file,
                "-v",
                "-s",
                "--tb=short"
            ]
            
            if self.quick_mode:
                cmd.append("-k")
                cmd.append("not extended")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            elapsed_time = time.time() - start_time
            
            # Parse results
            passed = result.returncode == 0
            
            self.results[suite_name] = {
                'passed': passed,
                'elapsed_time': elapsed_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
            if passed:
                logger.info(f"✓ {suite_name} PASSED ({elapsed_time:.2f}s)")
            else:
                logger.warning(f"✗ {suite_name} FAILED ({elapsed_time:.2f}s)")
            
            return passed
            
        except subprocess.TimeoutExpired:
            logger.error(f"✗ {suite_name} TIMEOUT")
            self.results[suite_name] = {
                'passed': False,
                'elapsed_time': 300,
                'error': 'Timeout'
            }
            return False
            
        except Exception as e:
            logger.error(f"✗ {suite_name} ERROR: {e}")
            self.results[suite_name] = {
                'passed': False,
                'elapsed_time': 0,
                'error': str(e)
            }
            return False
    
    def run_all_validations(self):
        """Run all validation test suites"""
        logger.info("\n" + "="*60)
        logger.info("SENTINEL SYSTEM VALIDATION SUITE")
        logger.info("="*60)
        
        if self.quick_mode:
            logger.info("Running in QUICK mode (skipping extended tests)")
        
        # Define test suites
        test_suites = [
            ("tests/test_integration_e2e.py", "End-to-End Integration"),
            ("tests/test_performance_validation.py", "Performance Validation"),
            ("tests/test_reliability_validation.py", "Reliability Validation"),
            ("tests/test_accuracy_validation.py", "Accuracy Validation")
        ]
        
        # Run each suite
        all_passed = True
        for test_file, suite_name in test_suites:
            test_path = Path(test_file)
            
            if not test_path.exists():
                logger.warning(f"⚠ {suite_name}: Test file not found: {test_file}")
                self.results[suite_name] = {
                    'passed': False,
                    'error': 'Test file not found'
                }
                all_passed = False
                continue
            
            passed = self.run_test_suite(test_file, suite_name)
            if not passed:
                all_passed = False
        
        return all_passed
    
    def generate_report(self):
        """Generate validation summary report"""
        report_lines = []
        
        report_lines.append("="*60)
        report_lines.append("SENTINEL SYSTEM VALIDATION REPORT")
        report_lines.append("="*60)
        report_lines.append("")
        
        # Summary
        total_suites = len(self.results)
        passed_suites = sum(1 for r in self.results.values() if r.get('passed', False))
        
        report_lines.append(f"Total Test Suites: {total_suites}")
        report_lines.append(f"Passed: {passed_suites}")
        report_lines.append(f"Failed: {total_suites - passed_suites}")
        report_lines.append("")
        
        # Individual results
        report_lines.append("DETAILED RESULTS:")
        report_lines.append("-"*60)
        
        for suite_name, result in self.results.items():
            status = "✓ PASS" if result.get('passed', False) else "✗ FAIL"
            elapsed = result.get('elapsed_time', 0)
            
            report_lines.append(f"\n{suite_name}:")
            report_lines.append(f"  Status: {status}")
            report_lines.append(f"  Time: {elapsed:.2f}s")
            
            if 'error' in result:
                report_lines.append(f"  Error: {result['error']}")
        
        report_lines.append("")
        report_lines.append("="*60)
        
        # Overall status
        if passed_suites == total_suites:
            report_lines.append("OVERALL STATUS: ✓ ALL VALIDATIONS PASSED")
        else:
            report_lines.append(f"OVERALL STATUS: ✗ {total_suites - passed_suites} VALIDATION(S) FAILED")
        
        report_lines.append("="*60)
        
        report_text = "\n".join(report_lines)
        
        # Print to console
        print("\n" + report_text)
        
        # Save to file if specified
        if self.report_file:
            report_path = Path(self.report_file)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report_text)
            logger.info(f"\nReport saved to: {self.report_file}")
        
        return report_text


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run SENTINEL system validation suite"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (skip extended tests)"
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Save report to file"
    )
    
    args = parser.parse_args()
    
    # Create validation suite
    suite = ValidationSuite(
        quick_mode=args.quick,
        report_file=args.report
    )
    
    # Run all validations
    start_time = time.time()
    all_passed = suite.run_all_validations()
    total_time = time.time() - start_time
    
    # Generate report
    suite.generate_report()
    
    logger.info(f"\nTotal validation time: {total_time:.2f}s")
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
