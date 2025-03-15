"""
LlamaIndex compatibility layer that provides consistent imports
across different versions of LlamaIndex.
"""

import importlib.util
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Try to import from different possible locations
def safe_import(name, alternative_names=None):
    if alternative_names is None:
        alternative_names = []
    
    # Try the primary name first
    try:
        return importlib.import_module(name)
    except ImportError:
        # Try alternatives
        for alt_name in alternative_names:
            try:
                return importlib.import_module(alt_name)
            except ImportError:
                continue
    
    # If we get here, nothing worked
    raise ImportError(f"Could not import {name} or any alternatives: {alternative_names}")

# Core components
core = safe_import("llama_index.core", ["llama_index"])
Document = getattr(core, "Document")
SimpleDirectoryReader = getattr(core, "SimpleDirectoryReader", None)
StorageContext = getattr(core, "StorageContext")
load_index_from_storage = getattr(core, "load_index_from_storage")

# Try to import Settings (new) or ServiceContext (old)
try:
    settings_module = safe_import("llama_index.settings", ["llama_index.core.settings"])
    Settings = getattr(settings_module, "Settings", None)
    ServiceContext = None  # We'll use Settings instead
except ImportError:
    # Fall back to ServiceContext
    ServiceContext = getattr(core, "ServiceContext", None)
    Settings = None

# Try to import VectorStoreIndex from different possible locations
try:
    indices = safe_import("llama_index.core.indices", ["llama_index.indices", "llama_index"])
    VectorStoreIndex = getattr(indices, "VectorStoreIndex", None)
except ImportError:
    try:
        vector_stores = safe_import("llama_index.vector_stores.types", ["llama_index.vector_stores", "llama_index"])
        VectorStoreIndex = getattr(vector_stores, "VectorStoreIndex", None)
    except ImportError:
        VectorStoreIndex = None

# If still not found, try direct import
if VectorStoreIndex is None:
    try:
        from llama_index import VectorStoreIndex
    except ImportError:
        # Last resort - we'll need to handle this case in the code that uses this
        VectorStoreIndex = None

# Node parser
try:
    node_parser_module = safe_import("llama_index.core.node_parser", ["llama_index.node_parser"])
    SimpleNodeParser = getattr(node_parser_module, "SimpleNodeParser", None)
except ImportError:
    SimpleNodeParser = None

# File readers
try:
    file_readers = safe_import("llama_index.readers.file", ["llama_index.core.readers.file"])
    PDFReader = getattr(file_readers, "PDFReader", None)
    DocxReader = getattr(file_readers, "DocxReader", None)
    CSVReader = getattr(file_readers, "CSVReader", None)
except ImportError:
    PDFReader = None
    DocxReader = None
    CSVReader = None

# Web readers
try:
    web_readers = safe_import("llama_index.readers.web", ["llama_index.core.readers.web"])
    SimpleWebPageReader = getattr(web_readers, "SimpleWebPageReader", None)
except ImportError:
    SimpleWebPageReader = None

# Create compatibility wrappers
class ServiceContextCompat:
    """Compatibility wrapper for ServiceContext/Settings"""
    
    @staticmethod
    def from_defaults(**kwargs):
        """Create a service context or configure settings based on availability"""
        if Settings is not None:
            # Use the new Settings
            try:
                # Check if we can directly set attributes on Settings
                for key, value in kwargs.items():
                    if hasattr(Settings, key):
                        setattr(Settings, key, value)
                return None  # No need to return anything when using Settings
            except Exception as e:
                print(f"Warning: Could not update Settings with kwargs: {e}")
                return None
        elif ServiceContext is not None:
            # Use the old ServiceContext
            return ServiceContext.from_defaults(**kwargs)
        else:
            raise ImportError("Neither Settings nor ServiceContext is available")

# Export the compatibility wrapper
service_context_compat = ServiceContextCompat()

# Check if we found everything we need
essential_components = [Document, SimpleDirectoryReader, StorageContext, load_index_from_storage,
                       VectorStoreIndex, SimpleNodeParser, PDFReader, DocxReader, CSVReader, SimpleWebPageReader]
if None in essential_components and (Settings is None and ServiceContext is None):
    missing = []
    for name, obj in [
        ("Document", Document),
        ("SimpleDirectoryReader", SimpleDirectoryReader),
        ("StorageContext", StorageContext),
        ("load_index_from_storage", load_index_from_storage),
        ("VectorStoreIndex", VectorStoreIndex),
        ("SimpleNodeParser", SimpleNodeParser),
        ("PDFReader", PDFReader),
        ("DocxReader", DocxReader),
        ("CSVReader", CSVReader),
        ("SimpleWebPageReader", SimpleWebPageReader),
        ("Settings/ServiceContext", Settings or ServiceContext)
    ]:
        if obj is None:
            missing.append(name)
    
    print(f"Warning: Could not import the following components: {', '.join(missing)}")
    print("Some functionality may not work properly.") 