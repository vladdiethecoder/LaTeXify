import re
from typing import Dict, Optional

class IdentifierMapper:
    """
    Handles symbol disambiguation and formatting.
    """
    
    GREEK_LETTERS = {
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
        'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho',
        'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
        'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta',
        'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Omicron', 'Pi', 'Rho',
        'Sigma', 'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega'
    }

    def __init__(self, overrides: Optional[Dict[str, str]] = None):
        self.overrides = overrides or {}

    def map_identifier(self, name: str) -> str:
        if name in self.overrides:
            return self.overrides[name]
        
        # Check for Greek letters
        if name in self.GREEK_LETTERS:
            return f"\\{name}"
            
        # Check for subscript pattern: var_sub
        # Only supports single underscore for now for simplicity, or x_y_z -> x_{y\_z}? 
        # Prompt says x_i -> x_{i}.
        if '_' in name:
            parts = name.split('_', 1)
            base = self.map_identifier(parts[0])
            sub = parts[1]
            # Recursively map the subscript part? Or just escape it?
            # For x_i, sub is i. If sub is alpha, it should be \alpha.
            # But if sub is a complex identifier?
            # Let's simple-recurse if it looks like an identifier.
            return f"{base}_{{{self.map_identifier(sub)}}}"
            
        return self._escape_latex(name)

    def _escape_latex(self, text: str) -> str:
        # Basic escaping for standard text that might end up in math mode? 
        # Actually in math mode, multi-character identifiers usually need \mathrm or similar if they are words.
        # But commonly in these tools, single chars are italics, multi chars are kept as is or wrapped.
        # For strict compliance, maybe \text{}?
        # Let's assume standard math italic for now unless it conflicts.
        if len(text) > 1:
            return f"\\mathrm{{{text}}}"
        return text
