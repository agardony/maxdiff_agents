"""
Tests for multi-persona functionality.
"""
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.main import load_personas
from src.types import ModelResponse


class TestLoadPersonas:
    """Test the load_personas function."""
    
    def test_single_persona_parameter_priority(self):
        """Test that single persona parameter takes priority over everything."""
        single_persona = "You are a test persona"
        personas = load_personas(None, single_persona)
        
        assert len(personas) == 1
        assert personas[0] == single_persona
    
    def test_single_persona_overrides_file(self):
        """Test that single persona parameter overrides personas file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("=== PERSONA 1 ===\nFile persona")
            temp_file = f.name
        
        try:
            single_persona = "Override persona"
            personas = load_personas(temp_file, single_persona)
            
            assert len(personas) == 1
            assert personas[0] == single_persona
        finally:
            os.unlink(temp_file)
    
    def test_valid_personas_file(self):
        """Test loading a valid personas file with multiple personas."""
        content = """=== PERSONA 1 ===
First persona content
with multiple lines

=== PERSONA 2 ===
Second persona content
also with multiple lines

=== PERSONA 3 ===
Third persona"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            personas = load_personas(temp_file, None)
            
            assert len(personas) == 3
            assert "First persona content" in personas[0]
            assert "multiple lines" in personas[0]
            assert "Second persona content" in personas[1]
            assert personas[2].strip() == "Third persona"
        finally:
            os.unlink(temp_file)
    
    def test_personas_file_case_insensitive_delimiters(self):
        """Test that delimiters are case insensitive."""
        content = """=== persona 1 ===
First persona

=== PERSONA 2 ===
Second persona

===  Persona  3  ===
Third persona"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            personas = load_personas(temp_file, None)
            
            assert len(personas) == 3
            assert personas[0].strip() == "First persona"
            assert personas[1].strip() == "Second persona"
            assert personas[2].strip() == "Third persona"
        finally:
            os.unlink(temp_file)
    
    def test_personas_file_single_persona(self):
        """Test loading a file with only one persona."""
        content = """=== PERSONA 1 ===
Only one persona here"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            personas = load_personas(temp_file, None)
            
            assert len(personas) == 1
            assert personas[0].strip() == "Only one persona here"
        finally:
            os.unlink(temp_file)
    
    def test_personas_file_no_delimiters(self):
        """Test file with no delimiters - should treat entire file as single persona."""
        content = """This is a persona without delimiters.
It spans multiple lines.
Should be treated as a single persona."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            personas = load_personas(temp_file, None)
            
            assert len(personas) == 1
            assert "This is a persona without delimiters" in personas[0]
            assert "Should be treated as a single persona" in personas[0]
        finally:
            os.unlink(temp_file)
    
    def test_personas_file_empty_file(self):
        """Test empty personas file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")
            temp_file = f.name
        
        try:
            personas = load_personas(temp_file, None)
            
            assert len(personas) == 0
        finally:
            os.unlink(temp_file)
    
    def test_personas_file_empty_personas(self):
        """Test file with delimiters but empty persona content."""
        content = """=== PERSONA 1 ===

=== PERSONA 2 ===

=== PERSONA 3 ===
Valid persona"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            personas = load_personas(temp_file, None)
            
            # Should only get the valid persona
            assert len(personas) == 1
            assert personas[0].strip() == "Valid persona"
        finally:
            os.unlink(temp_file)
    
    def test_personas_file_malformed_delimiters(self):
        """Test file with some malformed delimiters."""
        content = """=== PERSONA 1 ===
Valid persona 1

=== INVALID DELIMITER ===
This should not be recognized as a delimiter

=== PERSONA 2 ===
Valid persona 2"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            personas = load_personas(temp_file, None)
            
            assert len(personas) == 2
            assert "Valid persona 1" in personas[0]
            assert "This should not be recognized" in personas[0]  # Part of first persona
            assert personas[1].strip() == "Valid persona 2"
        finally:
            os.unlink(temp_file)
    
    def test_personas_file_not_found(self):
        """Test handling of non-existent personas file."""
        personas = load_personas("nonexistent_file.txt", None)
        
        assert len(personas) == 0
    
    def test_personas_file_whitespace_handling(self):
        """Test proper handling of whitespace around personas."""
        content = """=== PERSONA 1 ===


    Persona with leading/trailing whitespace    


=== PERSONA 2 ===
   Another persona   """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            personas = load_personas(temp_file, None)
            
            assert len(personas) == 2
            assert personas[0].strip() == "Persona with leading/trailing whitespace"
            assert personas[1].strip() == "Another persona"
        finally:
            os.unlink(temp_file)
    
    @patch.dict(os.environ, {'PERSONA': 'Env persona'})
    def test_fallback_to_env_persona(self):
        """Test fallback to environment PERSONA variable."""
        personas = load_personas(None, None)
        
        assert len(personas) == 1
        assert personas[0] == "Env persona"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_no_personas_found(self):
        """Test when no personas are available anywhere."""
        personas = load_personas(None, None)
        
        assert len(personas) == 0


class TestPersonaIntegration:
    """Test integration of personas with the rest of the system."""
    
    def test_model_response_persona_index(self):
        """Test that ModelResponse can handle persona_index."""
        response = ModelResponse(
            model_name="test-model",
            trial_number=1,
            best_item_id="item1",
            worst_item_id="item2",
            persona_index=2
        )
        
        assert response.persona_index == 2
        assert response.model_name == "test-model"
    
    def test_model_response_default_persona_index(self):
        """Test that ModelResponse handles missing persona_index."""
        response = ModelResponse(
            model_name="test-model",
            trial_number=1,
            best_item_id="item1",
            worst_item_id="item2"
        )
        
        assert response.persona_index is None


class TestPersonasFileFormats:
    """Test various personas file formats and edge cases."""
    
    def test_numbered_delimiters(self):
        """Test that numbered delimiters work correctly."""
        content = """=== PERSONA 1 ===
First
=== PERSONA 2 ===
Second
=== PERSONA 10 ===
Tenth"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            personas = load_personas(temp_file, None)
            
            assert len(personas) == 3
            assert personas[0].strip() == "First"
            assert personas[1].strip() == "Second"
            assert personas[2].strip() == "Tenth"
        finally:
            os.unlink(temp_file)
    
    def test_unicode_content(self):
        """Test personas with unicode characters."""
        content = """=== PERSONA 1 ===
You are evaluating café items with naïve enthusiasm 🍰

=== PERSONA 2 ===
你是一个美食评论家 (You are a food critic)"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_file = f.name
        
        try:
            personas = load_personas(temp_file, None)
            
            assert len(personas) == 2
            assert "café" in personas[0]
            assert "🍰" in personas[0]
            assert "美食评论家" in personas[1]
        finally:
            os.unlink(temp_file)
    
    def test_very_long_personas(self):
        """Test handling of very long persona descriptions."""
        long_persona = "You are a persona. " * 100  # 2000 characters
        content = f"""=== PERSONA 1 ===
{long_persona}

=== PERSONA 2 ===
Short persona"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            personas = load_personas(temp_file, None)
            
            assert len(personas) == 2
            assert len(personas[0]) > 1500
            assert "You are a persona." in personas[0]
            assert personas[1].strip() == "Short persona"
        finally:
            os.unlink(temp_file)


class TestErrorHandling:
    """Test error handling in persona loading."""
    
    def test_permission_denied_file(self):
        """Test handling of permission denied errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("=== PERSONA 1 ===\nTest persona")
            temp_file = f.name
        
        try:
            # Change permissions to make file unreadable
            os.chmod(temp_file, 0o000)
            
            personas = load_personas(temp_file, None)
            
            # Should return empty list on permission error
            assert len(personas) == 0
        finally:
            # Restore permissions and cleanup
            os.chmod(temp_file, 0o644)
            os.unlink(temp_file)
    
    def test_corrupted_file_handling(self):
        """Test handling of corrupted or invalid file content."""
        # Create a file with null bytes (common file corruption)
        content = "=== PERSONA 1 ===\nPersona with \x00 null bytes"
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
            f.write(content.encode('utf-8'))
            temp_file = f.name
        
        try:
            personas = load_personas(temp_file, None)
            
            # Should handle gracefully (may return empty or filtered content)
            assert isinstance(personas, list)
        finally:
            os.unlink(temp_file)
