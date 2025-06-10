#!/usr/bin/env python3
"""
Unit tests for UH loading to ensure regression prevention
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hrrr_processor_refactored import HRRRProcessor

def test_uh_loader_function_exists():
    """Test that the load_uh_layer function exists and handles errors gracefully"""
    
    processor = HRRRProcessor()
    
    # Test with non-existent file - should return None gracefully
    fake_path = Path("/tmp/nonexistent_file.grib2")
    result = processor.load_uh_layer(fake_path, 3000, 0)
    
    assert result is None, "load_uh_layer should return None for non-existent files"
    print("✅ load_uh_layer handles missing files correctly")

def test_uh_field_registry():
    """Test that UH fields are properly configured in the registry"""
    
    from field_registry import FieldRegistry
    
    registry = FieldRegistry()
    all_configs = registry.load_all_fields()
    
    # Check that uh_0_3 exists and has correct category
    assert 'uh_0_3' in all_configs, "uh_0_3 field should exist in registry"
    
    uh_0_3_config = all_configs['uh_0_3']
    assert uh_0_3_config.get('category') == 'updraft_helicity', \
        f"uh_0_3 should have category 'updraft_helicity', got {uh_0_3_config.get('category')}"
    
    assert 'layer' in uh_0_3_config, "uh_0_3 should have layer specification"
    assert uh_0_3_config['layer'] == '3000-0', \
        f"uh_0_3 should have layer '3000-0', got {uh_0_3_config['layer']}"
    
    # Check that uh_2_5 exists and has correct category
    assert 'uh_2_5' in all_configs, "uh_2_5 field should exist in registry"
    
    uh_2_5_config = all_configs['uh_2_5']
    assert uh_2_5_config.get('category') == 'updraft_helicity', \
        f"uh_2_5 should have category 'updraft_helicity', got {uh_2_5_config.get('category')}"
    
    assert 'layer' in uh_2_5_config, "uh_2_5 should have layer specification"
    assert uh_2_5_config['layer'] == '5000-2000', \
        f"uh_2_5 should have layer '5000-2000', got {uh_2_5_config['layer']}"
    
    print("✅ UH field registry entries are correctly configured")

def test_uh_category_detection():
    """Test that UH fields are detected by category for download triggering"""
    
    from field_registry import FieldRegistry
    
    registry = FieldRegistry()
    all_configs = registry.load_all_fields()
    
    # Filter to just UH fields
    uh_fields = {name: cfg for name, cfg in all_configs.items() 
                 if cfg.get('category') == 'updraft_helicity'}
    
    # Test the logic from the processor
    needs_uh = any(cfg.get('category') == 'updraft_helicity' for cfg in uh_fields.values())
    
    assert needs_uh, "UH category detection should trigger when UH fields are present"
    assert len(uh_fields) >= 2, f"Should have at least 2 UH fields, found {len(uh_fields)}"
    
    print(f"✅ Found {len(uh_fields)} UH fields with correct category detection")

if __name__ == "__main__":
    test_uh_loader_function_exists()
    test_uh_field_registry()
    test_uh_category_detection()
    print("✅ All UH regression tests passed!")