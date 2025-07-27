# config/secure_config.py - Secure configuration management
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class SecureConfig:
    """Secure configuration class using environment variables"""
    
    # LLM Provider API Keys (from environment)
    OPENAI_API_KEY: Optional[str] = None
    CLAUDE_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    
    # Model configurations
    OPENAI_MODEL: str = "gpt-4o-mini"
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"
    GEMINI_MODEL: str = "gemini-1.5-flash"
    
    # Development settings
    USE_MOCK_LLM: bool = False
    CACHE_ENABLED: bool = True
    CACHE_DIR: str = "cache"
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE: int = 60
    DAILY_REQUEST_LIMIT: int = 1500
    
    def __post_init__(self):
        """Load configuration from environment variables"""
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')  
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        
        # Development mode check
        self.USE_MOCK_LLM = os.getenv('USE_MOCK_LLM', 'false').lower() == 'true'
        
        # Override models if specified in environment
        self.OPENAI_MODEL = os.getenv('OPENAI_MODEL', self.OPENAI_MODEL)
        self.CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', self.CLAUDE_MODEL)
        self.GEMINI_MODEL = os.getenv('GEMINI_MODEL', self.GEMINI_MODEL)
        
        # Cache settings
        self.CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        self.CACHE_DIR = os.getenv('CACHE_DIR', self.CACHE_DIR)
    
    def validate_config(self) -> bool:
        """Validate that at least one API key is configured"""
        if self.USE_MOCK_LLM:
            return True
        
        api_keys = [
            self.OPENAI_API_KEY,
            self.CLAUDE_API_KEY, 
            self.GOOGLE_API_KEY
        ]
        
        return any(key for key in api_keys if key and key.strip())
    
    def get_available_providers(self) -> list:
        """Get list of available providers based on configured API keys"""
        providers = []
        
        if self.OPENAI_API_KEY:
            providers.append('openai')
        if self.CLAUDE_API_KEY:
            providers.append('claude')
        if self.GOOGLE_API_KEY:
            providers.append('gemini')
        
        if not providers or self.USE_MOCK_LLM:
            providers.append('mock')
            
        return providers

# Global configuration instance
config = SecureConfig()# Clean config without secrets
