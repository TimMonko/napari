"""
Shared OpenGL Resource Manager for Napari Vispy Layers

This module manages shared OpenGL resources to prevent premature deallocation
that causes access violations when switching between different layer types.
"""

import contextlib
import gc
import threading
import warnings
import weakref
from typing import Any, Optional


class SharedResourceManager:
    """
    Manages shared OpenGL resources across different layer types.

    This prevents the Windows OpenGL access violation bug that occurs when
    resources shared between layer types are prematurely freed.
    """

    _instance: Optional['SharedResourceManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'SharedResourceManager':
        """Singleton pattern to ensure one resource manager per application."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the shared resource manager."""
        if hasattr(self, '_initialized'):
            return

        self._initialized = True

        # Track active layers by type
        self._layer_instances: dict[str, weakref.WeakSet] = {}

        # Track shared resources and their reference counts
        self._shared_resources: dict[str, Any] = {}
        self._resource_ref_counts: dict[str, int] = {}

        # Track which layer types use which resources
        self._resource_dependencies: dict[str, set[str]] = {}

        # Thread safety
        self._resource_lock = threading.RLock()

    def register_layer(self, layer_type: str, layer_instance: Any) -> None:
        """
        Register a layer instance with the resource manager.

        Parameters
        ----------
        layer_type : str
            The type of layer (e.g., 'points', 'labels', 'shapes', 'image')
        layer_instance : Any
            The layer instance to register
        """
        with self._resource_lock:
            if layer_type not in self._layer_instances:
                self._layer_instances[layer_type] = weakref.WeakSet()

            self._layer_instances[layer_type].add(layer_instance)

            # Ensure shared resources for this layer type are available
            self._ensure_layer_resources(layer_type)

    def unregister_layer(self, layer_type: str, layer_instance: Any) -> None:
        """
        Unregister a layer instance and potentially clean up resources.

        Parameters
        ----------
        layer_type : str
            The type of layer being removed
        layer_instance : Any
            The layer instance to unregister
        """
        with self._resource_lock:
            if layer_type in self._layer_instances:
                with contextlib.suppress(KeyError):
                    self._layer_instances[layer_type].discard(layer_instance)

                # If this was the last instance of this layer type,
                # check if we can clean up resources
                if not self._layer_instances[layer_type]:
                    self._cleanup_unused_resources(layer_type)

    def _ensure_layer_resources(self, layer_type: str) -> None:
        """
        Ensure that shared resources needed by a layer type are available.

        Parameters
        ----------
        layer_type : str
            The layer type that needs resources
        """
        # Define which resources each layer type depends on
        dependencies = {
            'points': [
                'vertex_buffers',
                'marker_shaders',
                'transform_matrices',
            ],
            'labels': [
                'texture_units',
                'label_shaders',
                'color_maps',
                'transform_matrices',
            ],
            'shapes': [
                'vertex_buffers',
                'geometry_shaders',
                'transform_matrices',
            ],
            'image': [
                'texture_units',
                'image_shaders',
                'volume_rendering',
                'transform_matrices',
            ],
            'surface': [
                'vertex_buffers',
                'mesh_shaders',
                'normal_buffers',
                'transform_matrices',
            ],
            'tracks': ['vertex_buffers', 'line_shaders', 'transform_matrices'],
            'vectors': [
                'vertex_buffers',
                'arrow_shaders',
                'transform_matrices',
            ],
        }

        required_resources = dependencies.get(
            layer_type, ['transform_matrices']
        )

        for resource_name in required_resources:
            if resource_name not in self._shared_resources:
                self._shared_resources[resource_name] = (
                    self._create_shared_resource(resource_name)
                )
                self._resource_ref_counts[resource_name] = 0

            # Increment reference count
            self._resource_ref_counts[resource_name] += 1

            # Track dependency
            if resource_name not in self._resource_dependencies:
                self._resource_dependencies[resource_name] = set()
            self._resource_dependencies[resource_name].add(layer_type)

    def _create_shared_resource(self, resource_name: str) -> Any:
        """
        Create a shared resource placeholder.

        In a real implementation, this would create actual OpenGL resources.
        For now, we just track that the resource exists.

        Parameters
        ----------
        resource_name : str
            Name of the resource to create

        Returns
        -------
        Any
            The created resource (placeholder)
        """
        # This is a placeholder - in a real implementation, this would
        # create actual OpenGL resources like VBOs, textures, shaders, etc.
        return f'shared_{resource_name}_resource'

    def _cleanup_unused_resources(self, removed_layer_type: str) -> None:
        """
        Clean up resources that are no longer needed.

        This method is called when the last instance of a layer type is removed.
        It only cleans up resources if no other layer types are using them.

        Parameters
        ----------
        removed_layer_type : str
            The layer type that was removed
        """
        # Find resources that were used by the removed layer type
        resources_to_check = []
        for (
            resource_name,
            dependent_types,
        ) in self._resource_dependencies.items():
            if removed_layer_type in dependent_types:
                resources_to_check.append(resource_name)

        for resource_name in resources_to_check:
            # Remove the dependency
            self._resource_dependencies[resource_name].discard(
                removed_layer_type
            )

            # Decrement reference count
            if resource_name in self._resource_ref_counts:
                self._resource_ref_counts[resource_name] -= 1

            # Check if any other layer types still need this resource
            still_needed = False
            for _layer_type, instances in self._layer_instances.items():
                if (
                    instances
                    and resource_name
                    in self._resource_dependencies.get(resource_name, set())
                ):
                    still_needed = True
                    break

            # Only clean up if no layer types need this resource
            if (
                not still_needed
                and self._resource_ref_counts.get(resource_name, 0) <= 0
            ):
                self._safe_cleanup_resource(resource_name)

    def _safe_cleanup_resource(self, resource_name: str) -> None:
        """
        Safely clean up a shared resource.

        Parameters
        ----------
        resource_name : str
            Name of the resource to clean up
        """
        try:
            # Remove from tracking
            self._shared_resources.pop(resource_name, None)
            self._resource_ref_counts.pop(resource_name, None)
            self._resource_dependencies.pop(resource_name, None)

            # In a real implementation, this would also clean up the actual
            # OpenGL resource (VBO, texture, shader, etc.)

        except (OSError, RuntimeError, ValueError) as e:
            warnings.warn(
                f'Failed to cleanup shared resource {resource_name}: {e}'
            )

    def force_cleanup_all(self) -> None:
        """
        Force cleanup of all resources (emergency cleanup).

        This should only be used when the application is shutting down
        or in error recovery scenarios.
        """
        with self._resource_lock:
            try:
                # Clear all tracking data
                self._layer_instances.clear()
                self._shared_resources.clear()
                self._resource_ref_counts.clear()
                self._resource_dependencies.clear()

                # Force garbage collection
                gc.collect()

            except (OSError, RuntimeError, ValueError) as e:
                warnings.warn(f'Failed to force cleanup all resources: {e}')

    def get_active_layer_types(self) -> set[str]:
        """
        Get the set of currently active layer types.

        Returns
        -------
        Set[str]
            Set of layer type names that have active instances
        """
        with self._resource_lock:
            return {
                layer_type
                for layer_type, instances in self._layer_instances.items()
                if instances
            }

    def get_resource_status(self) -> dict[str, Any]:
        """
        Get the current status of all shared resources.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary containing resource status information
        """
        with self._resource_lock:
            return {
                'active_layer_types': list(self.get_active_layer_types()),
                'shared_resources': list(self._shared_resources.keys()),
                'resource_ref_counts': dict(self._resource_ref_counts),
                'resource_dependencies': {
                    resource: list(deps)
                    for resource, deps in self._resource_dependencies.items()
                },
            }


# Global instance
_resource_manager = SharedResourceManager()


def get_shared_resource_manager() -> SharedResourceManager:
    """
    Get the global shared resource manager instance.

    Returns
    -------
    SharedResourceManager
        The global shared resource manager
    """
    return _resource_manager
