"""
LLM Verification module for confidence-based hybrid pipeline.
Uses Groq API (free) for fact-checking low-confidence predictions.
"""

import os
import json
from typing import Dict, Optional


def build_prompt(text: str, roberta_prediction: str, confidence: float) -> str:
    """Build fact-checking prompt for LLM."""
    return f"""You are a fact-checking AI. Analyze the following news article and determine if it is real or fake.

The RoBERTa classifier predicted this article as **{roberta_prediction}** with {confidence:.1%} confidence, but the confidence is too low to be reliable.

--- NEWS ARTICLE ---
{text[:2000]}
--- END ---

Provide your analysis in JSON format:
{{
    "reasoning": "Your step-by-step reasoning",
    "verdict": "real" or "fake",
    "confidence": 0.0-1.0,
    "key_indicators": ["indicator1", "indicator2"]
}}

Respond ONLY with the JSON object."""


def parse_response(response_text: str) -> Dict:
    """Parse LLM JSON response."""
    try:
        # Try direct JSON parse
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(response_text[start:end])
    except json.JSONDecodeError:
        pass

    return {
        "reasoning": response_text,
        "verdict": "uncertain",
        "confidence": 0.0,
        "key_indicators": []
    }


class LLMVerifier:
    """LLM-based fact checker using Groq API or Gemini API."""

    def __init__(self, groq_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None,
                 groq_model: str = "llama-3.3-70b-versatile", gemini_model: str = "gemini-2.5-flash",
                 max_tokens: int = 1024, temperature: float = 0.1):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.groq_model = groq_model
        self.gemini_model = gemini_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._groq_client = None

        if self.gemini_api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)

    @property
    def groq_client(self):
        if self._groq_client is None and self.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_api_key)
            except ImportError:
                pass
        return self._groq_client

    @property
    def is_available(self) -> bool:
        return self.groq_api_key is not None or self.gemini_api_key is not None

    def verify(self, text: str, roberta_prediction: str, confidence: float) -> Dict:
        """Send text to LLM for verification, trying Groq first, then Gemini."""
        if not self.is_available:
            return {"verdict": "unavailable", "reasoning": "No API keys configured"}

        prompt = build_prompt(text, roberta_prediction, confidence)

        # Try Groq
        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                return parse_response(response.choices[0].message.content)
            except Exception as e:
                pass # Fallback to Gemini if Groq fails

        # Try Gemini
        if self.gemini_api_key:
            try:
                import google.generativeai as genai
                model = genai.GenerativeModel(self.gemini_model)
                response = model.generate_content(
                    prompt, 
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature, 
                        max_output_tokens=self.max_tokens
                    )
                )
                if response.text:
                    return parse_response(response.text)
            except Exception as e:
                return {"verdict": "error", "reasoning": f"Gemini error: {str(e)}", "confidence": 0.0, "key_indicators": []}
                
        return {"verdict": "error", "reasoning": "All LLMs failed or not configured", "confidence": 0.0, "key_indicators": []}


def hybrid_predict(
    model, tokenizer, text: str, device,
    mc_dropout_fn, temp_scaler=None,
    llm_verifier: Optional[LLMVerifier] = None,
    confidence_threshold: float = 0.7,
    n_mc_runs: int = 10, max_length: int = 256
) -> Dict:
    """Full hybrid pipeline: RoBERTa → MC Dropout → Temp Scaling → LLM.

    Returns dict with prediction, confidence, uncertainty, and LLM result.
    """
    import torch

    # Step 1: Tokenize
    encoding = tokenizer(text, max_length=max_length, padding='max_length',
                         truncation=True, return_tensors='pt')

    # Step 2: MC Dropout prediction
    mc_result = mc_dropout_fn(
        model=model,
        input_ids=encoding['input_ids'],
        attention_mask=encoding['attention_mask'],
        n_runs=n_mc_runs, device=device
    )

    pred_idx = int(mc_result['predictions'][0])
    from .config import LABEL_MAP_INV
    prediction = LABEL_MAP_INV[pred_idx]
    confidence = float(mc_result['mean_probs'][0][pred_idx])
    uncertainty = float(mc_result['uncertainty'][0])

    # Step 3: Temperature Scaling (if available)
    if temp_scaler is not None:
        import numpy as np
        logits = torch.tensor(np.log(mc_result['mean_probs'] + 1e-10))
        calibrated = temp_scaler.calibrate(logits.to(device))
        calibrated = calibrated.cpu().numpy()
        confidence = float(calibrated[0][pred_idx])
        mc_result['mean_probs'] = calibrated

    result = {
        'prediction': prediction,
        'confidence': confidence,
        'uncertainty': uncertainty,
        'probabilities': {
            'real': float(mc_result['mean_probs'][0][0]),
            'fake': float(mc_result['mean_probs'][0][1])
        },
        'method': 'roberta_mc_dropout',
        'llm_verification': None
    }

    # Step 4: LLM verification if low confidence
    if confidence < confidence_threshold and llm_verifier and llm_verifier.is_available:
        print(f"\n[LLM_TRIGGER] Threshold met! {confidence:.4f} < {confidence_threshold}. Sending to Groq/Gemini...\n")
        llm_result = llm_verifier.verify(text, prediction, confidence)
        result['llm_verification'] = llm_result
        result['method'] = 'hybrid_with_llm'

        if llm_result.get('verdict') in ['real', 'fake']:
            result['prediction'] = llm_result['verdict']
            result['confidence'] = llm_result.get('confidence', confidence)

    return result
