"""
Smart Error Diagnosis
Analyzes errors and provides helpful suggestions for fixes
"""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ErrorDiagnoser:
    """Diagnose errors and suggest fixes"""

    def __init__(self):
        """Initialize error patterns and fixes"""
        self.error_patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> Dict[str, Dict]:
        """Initialize common error patterns and their fixes"""
        return {
            'ModuleNotFoundError': {
                'pattern': r"ModuleNotFoundError: No module named '(\w+)'",
                'category': 'dependency',
                'fixes': [
                    "Install the missing package: pip install {module}",
                    "Check if package name has changed or been deprecated",
                    "Try installing with conda: conda install {module}",
                    "Check requirements.txt for correct package name"
                ]
            },
            'ImportError': {
                'pattern': r"ImportError: cannot import name '(\w+)'",
                'category': 'dependency',
                'fixes': [
                    "Package version mismatch - check requirements.txt for correct version",
                    "Try upgrading the package: pip install --upgrade {module}",
                    "The import may have been moved in newer versions"
                ]
            },
            'CUDA_ERROR': {
                'pattern': r"CUDA|cuda|RuntimeError: CUDA",
                'category': 'gpu',
                'fixes': [
                    "Install CUDA toolkit matching PyTorch/TensorFlow version",
                    "Check GPU availability: nvidia-smi",
                    "Set environment to use CPU: export CUDA_VISIBLE_DEVICES=''",
                    "Update GPU drivers",
                    "Try CPU version of the framework"
                ]
            },
            'OutOfMemoryError': {
                'pattern': r"OutOfMemoryError|CUDA out of memory",
                'category': 'gpu',
                'fixes': [
                    "Reduce batch size in training/inference",
                    "Clear GPU cache: torch.cuda.empty_cache()",
                    "Use gradient accumulation instead of large batches",
                    "Enable mixed precision training (fp16)",
                    "Use a smaller model variant"
                ]
            },
            'FileNotFoundError': {
                'pattern': r"FileNotFoundError.*['\"](.+?)['\"]",
                'category': 'data',
                'fixes': [
                    "Download required data files",
                    "Check data directory paths in config files",
                    "Run data preparation scripts first",
                    "Update file paths to match your directory structure"
                ]
            },
            'PermissionError': {
                'pattern': r"PermissionError",
                'category': 'system',
                'fixes': [
                    "Check file/directory permissions: ls -la",
                    "You may need write permissions: chmod +w {file}",
                    "Try running with appropriate permissions",
                    "Check if file is being used by another process"
                ]
            },
            'ConnectionError': {
                'pattern': r"ConnectionError|Connection refused|timeout",
                'category': 'network',
                'fixes': [
                    "Check internet connection",
                    "API endpoint may be down - try again later",
                    "Check firewall settings",
                    "Increase timeout in configuration"
                ]
            },
            'ValueError': {
                'pattern': r"ValueError: (.+)",
                'category': 'runtime',
                'fixes': [
                    "Check input data format and types",
                    "Verify configuration parameters",
                    "Look at the specific error message for hints",
                    "Check data preprocessing steps"
                ]
            },
            'KeyError': {
                'pattern': r"KeyError: ['\"](\w+)['\"]",
                'category': 'runtime',
                'fixes': [
                    "Missing required key '{key}' in config or data",
                    "Check configuration file for required fields",
                    "Verify data format matches expected structure"
                ]
            },
            'TypeError': {
                'pattern': r"TypeError: (.+)",
                'category': 'runtime',
                'fixes': [
                    "Type mismatch - check function arguments",
                    "May be caused by version incompatibility",
                    "Review API documentation for correct types"
                ]
            },
            'VersionConflict': {
                'pattern': r"version|Version|compatible|compatibility",
                'category': 'dependency',
                'fixes': [
                    "Check package version requirements",
                    "Create fresh virtual environment",
                    "Use pip freeze to check installed versions",
                    "Refer to paper's requirements for exact versions"
                ]
            }
        }

    def diagnose(self, error_message: str, context: Optional[Dict] = None) -> Dict:
        """
        Diagnose an error and provide fixes

        Args:
            error_message: Error message/stack trace
            context: Additional context (env info, dependencies, etc.)

        Returns:
            Dict with diagnosis and suggested fixes
        """
        diagnosis = {
            'error_type': 'unknown',
            'category': 'unknown',
            'description': error_message[:200],
            'root_cause': None,
            'suggested_fixes': [],
            'matched_pattern': False
        }

        # Try to match error patterns
        for error_type, info in self.error_patterns.items():
            pattern = info['pattern']
            match = re.search(pattern, error_message, re.IGNORECASE | re.MULTILINE)

            if match:
                diagnosis['error_type'] = error_type
                diagnosis['category'] = info['category']
                diagnosis['matched_pattern'] = True

                # Extract specific details (like module name, key, etc.)
                if match.groups():
                    detail = match.group(1)
                    diagnosis['root_cause'] = detail

                    # Customize fixes with specific details
                    fixes = []
                    for fix in info['fixes']:
                        if '{module}' in fix:
                            fixes.append(fix.format(module=detail))
                        elif '{key}' in fix:
                            fixes.append(fix.format(key=detail))
                        elif '{file}' in fix:
                            fixes.append(fix.format(file=detail))
                        else:
                            fixes.append(fix)

                    diagnosis['suggested_fixes'] = fixes
                else:
                    diagnosis['suggested_fixes'] = info['fixes']

                break

        # Add context-specific suggestions
        if context:
            additional_fixes = self._get_context_specific_fixes(diagnosis, context)
            diagnosis['suggested_fixes'].extend(additional_fixes)

        return diagnosis

    def _get_context_specific_fixes(self, diagnosis: Dict, context: Dict) -> List[str]:
        """Get additional fixes based on context"""
        fixes = []

        # GPU-related context
        if diagnosis['category'] == 'gpu':
            if not context.get('gpu_available', True):
                fixes.append("⚠️ No GPU detected - consider using CPU-only version")

        # Dependency context
        if diagnosis['category'] == 'dependency':
            if context.get('python_version'):
                fixes.append(f"Python version: {context['python_version']} - check compatibility")

        # Environment context
        if context.get('docker_available'):
            fixes.append("💡 Consider using Docker for better isolation")

        return fixes

    def analyze_execution_result(self, result_dict: Dict) -> Dict:
        """
        Analyze full execution result

        Args:
            result_dict: ExecutionResult.to_dict()

        Returns:
            Comprehensive diagnosis
        """
        diagnosis = {
            'success': result_dict.get('success', False),
            'exit_code': result_dict.get('exit_code'),
            'execution_time': result_dict.get('execution_time', 0),
            'errors': [],
            'warnings': [],
            'recommendations': []
        }

        # Analyze errors
        if result_dict.get('errors'):
            for error in result_dict['errors']:
                error_diagnosis = self.diagnose(error)
                diagnosis['errors'].append(error_diagnosis)

        # Analyze stderr
        stderr = result_dict.get('stderr', '')
        if stderr:
            error_diagnosis = self.diagnose(stderr)
            if error_diagnosis['matched_pattern']:
                diagnosis['errors'].append(error_diagnosis)

        # Analyze warnings
        if result_dict.get('warnings'):
            diagnosis['warnings'] = result_dict['warnings']

        # Generate recommendations
        if not diagnosis['success']:
            diagnosis['recommendations'] = self._generate_recommendations(diagnosis)

        return diagnosis

    def _generate_recommendations(self, diagnosis: Dict) -> List[str]:
        """Generate overall recommendations based on diagnosis"""
        recommendations = []

        # Group errors by category
        categories = {}
        for error in diagnosis['errors']:
            cat = error.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        # Recommend based on most common category
        if categories:
            most_common = max(categories, key=categories.get)

            if most_common == 'dependency':
                recommendations.append("🔧 Consider creating a fresh virtual environment")
                recommendations.append("📦 Review and install all dependencies from requirements.txt")

            elif most_common == 'gpu':
                recommendations.append("🎮 Check GPU availability and CUDA installation")
                recommendations.append("💻 Consider running on CPU or cloud GPU service")

            elif most_common == 'data':
                recommendations.append("📁 Verify all required data files are downloaded")
                recommendations.append("🔍 Check README for data preparation instructions")

            elif most_common == 'network':
                recommendations.append("🌐 Check internet connectivity")
                recommendations.append("⏱️ Try increasing timeout settings")

        # General recommendations
        if diagnosis['exit_code'] != 0:
            recommendations.append("📝 Review full error logs for more details")

        return recommendations

    def format_diagnosis(self, diagnosis: Dict) -> str:
        """Format diagnosis for display"""
        output = []

        if diagnosis.get('error_type'):
            output.append(f"❌ Error Type: {diagnosis['error_type']}")

        if diagnosis.get('category'):
            output.append(f"📂 Category: {diagnosis['category']}")

        if diagnosis.get('root_cause'):
            output.append(f"🎯 Root Cause: {diagnosis['root_cause']}")

        if diagnosis.get('suggested_fixes'):
            output.append("\n🔧 Suggested Fixes:")
            for i, fix in enumerate(diagnosis['suggested_fixes'], 1):
                output.append(f"  {i}. {fix}")

        if diagnosis.get('recommendations'):
            output.append("\n💡 Recommendations:")
            for rec in diagnosis['recommendations']:
                output.append(f"  • {rec}")

        return '\n'.join(output)


# Global diagnoser instance
_diagnoser = None


def diagnose_error(error_message: str, context: Optional[Dict] = None) -> Dict:
    """Convenience function to diagnose an error"""
    global _diagnoser
    if _diagnoser is None:
        _diagnoser = ErrorDiagnoser()
    return _diagnoser.diagnose(error_message, context)


def get_quick_fix(error_type: str) -> List[str]:
    """Get quick fixes for common error types"""
    diagnoser = ErrorDiagnoser()
    pattern_info = diagnoser.error_patterns.get(error_type)

    if pattern_info:
        return pattern_info['fixes']

    return ["No quick fixes available for this error type"]
