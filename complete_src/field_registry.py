#!/usr/bin/env python3
"""
HRRR Field Registry
Central registry for managing and accessing field configurations
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import json
import copy

from field_templates import FieldTemplates


class FieldRegistry:
    """Central registry for HRRR field configurations"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize field registry
        
        Args:
            config_dir: Directory containing parameter configuration files
        """
        self.config_dir = config_dir or Path(__file__).parent / 'parameters'
        self.templates = FieldTemplates()
        self._fields = {}
        self._loaded = False
        self._field_cache = {}
        
    def load_all_fields(self, force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
        """Load all field configurations"""
        if self._loaded and not force_reload:
            return self._fields
            
        print("🔄 Loading field configurations...")
        
        # Load parameter definitions
        param_defs = self.load_all_parameters()
        if not param_defs:
            print("❌ No parameter definitions found")
            return {}
        
        # Build configurations
        self._fields = self.build_all_configs(param_defs)
        self._loaded = True
        
        if self._fields:
            print(f"✅ Loaded {len(self._fields)} field configurations")
            self._print_summary()
        else:
            print("❌ Failed to load field configurations")
            
        return self._fields
    
    def get_field(self, field_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for specific field"""
        if not self._loaded:
            self.load_all_fields()
        return self._fields.get(field_name)
    
    def get_all_fields(self) -> Dict[str, Dict[str, Any]]:
        """Get all field configurations"""
        if not self._loaded:
            self.load_all_fields()
        return self._fields.copy()
    
    def get_fields_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get all fields in specific category"""
        if not self._loaded:
            self.load_all_fields()
        return {name: config for name, config in self._fields.items() 
                if config.get('category') == category}
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories"""
        if not self._loaded:
            self.load_all_fields()
        categories = {config.get('category') for config in self._fields.values() 
                     if config.get('category')}
        return sorted(list(categories))
    
    def get_field_names(self, category: Optional[str] = None) -> List[str]:
        """Get list of field names, optionally filtered by category"""
        if category:
            fields = self.get_fields_by_category(category)
            return list(fields.keys())
        else:
            if not self._loaded:
                self.load_all_fields()
            return list(self._fields.keys())
    
    def field_exists(self, field_name: str) -> bool:
        """Check if field exists"""
        if not self._loaded:
            self.load_all_fields()
        return field_name in self._fields
    
    def add_field(self, field_name: str, field_config: Dict[str, Any], 
                  save_to_file: bool = False) -> bool:
        """Add new field configuration
        
        Args:
            field_name: Name of the field
            field_config: Field configuration dictionary
            save_to_file: Whether to save to a configuration file
        """
        try:
            # Build complete configuration
            complete_config = self.build_field_config(field_name, field_config)
            if complete_config:
                self._fields[field_name] = complete_config
                print(f"✅ Added field: {field_name}")
                
                if save_to_file:
                    self._save_field_to_file(field_name, field_config)
                
                return True
            else:
                print(f"❌ Failed to add field: {field_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error adding field {field_name}: {e}")
            return False
    
    def remove_field(self, field_name: str) -> bool:
        """Remove field from registry"""
        if field_name in self._fields:
            del self._fields[field_name]
            print(f"✅ Removed field: {field_name}")
            return True
        else:
            print(f"❌ Field not found: {field_name}")
            return False
    
    def validate_all_fields(self) -> bool:
        """Validate all field configurations"""
        if not self._loaded:
            self.load_all_fields()
        
        valid_count = 0
        for field_name, config in self._fields.items():
            if self.templates.validate_config(config):
                valid_count += 1
            else:
                print(f"❌ Invalid configuration: {field_name}")
        
        return valid_count == len(self._fields)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of loaded fields"""
        if not self._loaded:
            self.load_all_fields()
            
        # Count by category
        categories = {}
        for config in self._fields.values():
            category = config.get('category', 'uncategorized')
            categories[category] = categories.get(category, 0) + 1
        
        # Count by colormap
        colormaps = {}
        for config in self._fields.values():
            cmap = config.get('cmap', 'default')
            colormaps[cmap] = colormaps.get(cmap, 0) + 1
        
        return {
            'total_fields': len(self._fields),
            'categories': categories,
            'colormaps': colormaps
        }
    
    def search_fields(self, search_term: str, search_in: List[str] = None) -> List[str]:
        """Search for fields by name, title, or other attributes
        
        Args:
            search_term: Term to search for
            search_in: List of attributes to search in ['name', 'title', 'var', 'category']
        """
        if not self._loaded:
            self.load_all_fields()
            
        if search_in is None:
            search_in = ['name', 'title', 'var', 'category']
        
        results = []
        search_lower = search_term.lower()
        
        for field_name, config in self._fields.items():
            # Search in field name
            if 'name' in search_in and search_lower in field_name.lower():
                results.append(field_name)
                continue
                
            # Search in other attributes
            for attr in search_in:
                if attr != 'name' and attr in config:
                    if search_lower in str(config[attr]).lower():
                        results.append(field_name)
                        break
        
        return sorted(list(set(results)))
    
    def export_fields(self, output_file: Path, category: Optional[str] = None):
        """Export field configurations to file"""
        if not self._loaded:
            self.load_all_fields()
            
        if category:
            fields_to_export = self.get_fields_by_category(category)
        else:
            fields_to_export = self._fields
            
        # Export configurations to file
        with open(output_file, 'w') as f:
            json.dump(fields_to_export, f, indent=2)
        print(f"💾 Exported {len(fields_to_export)} configurations to {output_file}")
    
    def _save_field_to_file(self, field_name: str, field_config: Dict[str, Any]):
        """Save field configuration to appropriate category file"""
        category = field_config.get('category', 'custom')
        category_file = self.config_dir / f"{category}.json"
        
        # Load existing category file
        if category_file.exists():
            with open(category_file, 'r') as f:
                category_data = json.load(f)
        else:
            category_data = {}
        
        # Add new field
        category_data[field_name] = field_config
        
        # Save updated file
        with open(category_file, 'w') as f:
            json.dump(category_data, f, indent=2)
        
        print(f"💾 Saved {field_name} to {category_file}")
    
    def load_all_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Load all parameter configurations from config directory"""
        all_params = {}
        
        if not self.config_dir.exists():
            print(f"Config directory not found: {self.config_dir}")
            return {}
        
        # Load all configuration files
        for config_file in self.config_dir.glob('*.json'):
            file_params = self.load_parameter_file(config_file)
            if file_params:
                print(f"Loaded {len(file_params)} parameters from {config_file.name}")
                all_params.update(file_params)
                
        return all_params
    
    def load_parameter_file(self, file_path: Path) -> Dict[str, Any]:
        """Load parameter configuration from JSON file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {}
    
    def build_all_configs(self, param_defs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build configurations for all parameters"""
        configs = {}
        total_built = 0
        
        for field_name, field_def in param_defs.items():
            try:
                config = self.build_field_config(field_name, field_def)
                if config:
                    configs[field_name] = config
                    total_built += 1
            except Exception as e:
                print(f"Error building config for {field_name}: {e}")
        
        print(f"Successfully built {total_built} field configurations")
        return configs
    
    def build_field_config(self, field_name: str, field_def: Dict[str, Any]) -> Dict[str, Any]:
        """Build complete field configuration from definition"""
        try:
            # Use cached config if available
            cache_key = f"{field_name}:{hash(str(sorted(field_def.items())))}"
            if cache_key in self._field_cache:
                return copy.deepcopy(self._field_cache[cache_key])
            
            # Resolve template and build config
            resolved_config = self.templates.resolve_template(field_def)
            
            # Add field name if not present
            if 'name' not in resolved_config:
                resolved_config['name'] = field_name
            
            # Validate configuration
            if not self.templates.validate_config(resolved_config):
                raise ValueError(f"Invalid configuration for field: {field_name}")
            
            # Cache and return
            self._field_cache[cache_key] = copy.deepcopy(resolved_config)
            return resolved_config
            
        except Exception as e:
            print(f"Error building config for {field_name}: {e}")
            return None
    
    def _print_summary(self):
        """Print summary of loaded fields"""
        summary = self.get_summary()
        
        print(f"📊 Field Registry Summary:")
        print(f"  Total fields: {summary['total_fields']}")
        
        print(f"  Categories:")
        for category, count in sorted(summary['categories'].items()):
            print(f"    {category}: {count}")
        
        print(f"  Top colormaps:")
        for cmap, count in sorted(summary['colormaps'].items(), 
                                 key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {cmap}: {count}")


def create_registry_example():
    """Example of how to use the field registry"""
    registry = FieldRegistry()
    
    # Load all fields
    fields = registry.load_all_fields()
    
    # Get specific field
    cape_field = registry.get_field('sbcape')
    if cape_field:
        print(f"\nSBCAPE Configuration:")
        for key, value in cape_field.items():
            print(f"  {key}: {value}")
    
    # Get fields by category
    instability_fields = registry.get_fields_by_category('instability')
    print(f"\nInstability fields: {list(instability_fields.keys())}")
    
    # Search fields
    cape_fields = registry.search_fields('cape')
    print(f"\nFields containing 'cape': {cape_fields}")
    
    # Add new field
    new_field_config = {
        'template': 'surface_temperature',
        'param_id': 999,
        'var': 'test_temp',
        'title': 'Test Temperature'
    }
    registry.add_field('test_temperature', new_field_config)
    
    return registry


if __name__ == '__main__':
    # Demo the registry
    create_registry_example()