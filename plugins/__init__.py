# plugins/__init__.py - Plugin registry for discovering and loading plugins
from typing import Dict, Type, List
import importlib
import pkgutil
import inspect
import os
from .base import InferencePlugin

class PluginRegistry:
    """
    Registry for managing inference plugins
    """
    def __init__(self):
        self._plugins: Dict[str, Type[InferencePlugin]] = {}
        self._loaded_instances: Dict[str, Dict[str, InferencePlugin]] = {}
    
    def register_plugin(self, plugin_class: Type[InferencePlugin]) -> None:
        """Register a plugin class"""
        if not inspect.isclass(plugin_class) or not issubclass(plugin_class, InferencePlugin):
            raise TypeError("Plugin must be a subclass of InferencePlugin")
        
        plugin_name = plugin_class.__name__
        self._plugins[plugin_name] = plugin_class
    
    def get_plugin_class(self, name: str) -> Type[InferencePlugin]:
        """Get a plugin class by name"""
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' not found")
        return self._plugins[name]
    
    def list_plugins(self) -> List[str]:
        """List all registered plugins"""
        return list(self._plugins.keys())
    
    def load_plugin(self, name: str, model_id: str, model_path: str, **kwargs) -> InferencePlugin:
        """
        Load a specific model using the specified plugin
        Returns an instance of the plugin with the model loaded
        """
        plugin_class = self.get_plugin_class(name)
        plugin_instance = plugin_class()
        plugin_instance.load_model(model_path, **kwargs)
        
        # Store the instance for later use
        if name not in self._loaded_instances:
            self._loaded_instances[name] = {}
        
        self._loaded_instances[name][model_id] = plugin_instance
        return plugin_instance
    
    def get_instance(self, plugin_name: str, model_id: str) -> InferencePlugin:
        """Get a loaded plugin instance"""
        if plugin_name not in self._loaded_instances or model_id not in self._loaded_instances[plugin_name]:
            raise ValueError(f"Plugin '{plugin_name}' with model ID '{model_id}' not loaded")
        return self._loaded_instances[plugin_name][model_id]
    
    def discover_plugins(self, package='plugins'):
        """Auto-discover plugins in the plugins directory"""
        import plugins
        plugin_package = importlib.import_module(package)
        
        for _, name, is_pkg in pkgutil.iter_modules(plugin_package.__path__, plugin_package.__name__ + '.'):
            if not is_pkg and name != f"{package}.base":
                module = importlib.import_module(name)
                for item_name, item in inspect.getmembers(module, inspect.isclass):
                    if issubclass(item, InferencePlugin) and item != InferencePlugin:
                        self.register_plugin(item)
                        
# Create the registry instance
registry = PluginRegistry()