"""
Fake News Detection with RoBERTa
Multi-stage detection pipeline with uncertainty estimation and LLM verification.
"""

# Lazy imports — các module được import riêng trong notebook
# để tránh lỗi dependency chain trên Kaggle/Colab
__all__ = [
    'config', 'utils', 'dataset', 'model',
    'trainer', 'evaluation', 'calibration', 'llm_verifier'
]
